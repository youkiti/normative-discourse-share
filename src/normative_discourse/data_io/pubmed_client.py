"""PubMed E-utilities API client.

Handles query construction, ESearch, EFetch, and XML parsing.
NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any, Optional

try:  # only needed for the optional online abstract fallback (build-corpus
    import requests  # without a cached pubmed_abstracts.json). The documented
except ModuleNotFoundError:  # reproduction path uses the cache and never imports this.
    requests = None  # type: ignore[assignment]

from normative_discourse.schemas import ArticleRecord, JournalField

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# Max records per efetch batch (NCBI recommends ≤500)
EFETCH_BATCH_SIZE = 200


def build_journal_query(
    journal_abbreviation: str,
    date_start: str,
    date_end: str,
    publication_types: list[str],
    exclude_publication_types: list[str],
    language: str = "english",
    require_abstract: bool = True,
) -> str:
    """Build a PubMed query string for a single journal.

    Args:
        journal_abbreviation: NLM title abbreviation (e.g. "N Engl J Med")
        date_start: Start date "YYYY/MM/DD"
        date_end: End date "YYYY/MM/DD"
        publication_types: Inclusion types (e.g. ["Journal Article"])
        exclude_publication_types: Exclusion types
        language: Language filter
        require_abstract: Whether to require abstract presence

    Returns:
        PubMed query string
    """
    # Build inclusion part (joined with AND)
    incl_parts: list[str] = []

    # Journal (quoted for exact match on [ta] field)
    incl_parts.append(f'"{journal_abbreviation}"[ta]')

    # Date range
    incl_parts.append(f'("{date_start}"[dp] : "{date_end}"[dp])')

    # Inclusion publication types (OR)
    if publication_types:
        pt_incl = " OR ".join(f'"{pt}"[pt]' for pt in publication_types)
        incl_parts.append(f"({pt_incl})")

    # Language
    if language:
        incl_parts.append(f"{language}[la]")

    # Abstract
    if require_abstract:
        incl_parts.append("hasabstract")

    query = " AND ".join(incl_parts)

    # Exclusion publication types: use "NOT" (not "AND NOT")
    # PubMed interprets "A NOT B" as "A minus B"
    for pt in exclude_publication_types:
        query += f' NOT "{pt}"[pt]'

    return query


def esearch(
    query: str,
    email: str,
    api_key: str | None = None,
    api_delay: float = 0.34,
) -> dict[str, Any]:
    """Run ESearch and return WebEnv + QueryKey for history-based fetching.

    Returns:
        dict with keys: count, web_env, query_key, id_list
    """
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,  # We use history server, don't need IDs here
        "usehistory": "y",
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    time.sleep(api_delay)
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    count_el = root.find("Count")
    count = int(count_el.text) if count_el is not None and count_el.text else 0

    web_env_el = root.find("WebEnv")
    web_env = web_env_el.text if web_env_el is not None else ""

    query_key_el = root.find("QueryKey")
    query_key = query_key_el.text if query_key_el is not None else ""

    # Also get ID list for small result sets
    id_list_el = root.find("IdList")
    id_list: list[str] = []
    if id_list_el is not None:
        id_list = [
            id_el.text for id_el in id_list_el.findall("Id") if id_el.text
        ]

    logger.info("ESearch: %d results for query (truncated): %.80s...", count, query)

    return {
        "count": count,
        "web_env": web_env,
        "query_key": query_key,
        "id_list": id_list,
    }


def efetch_batch(
    web_env: str,
    query_key: str,
    total: int,
    email: str,
    api_key: str | None = None,
    batch_size: int = EFETCH_BATCH_SIZE,
    api_delay: float = 0.34,
) -> str:
    """Fetch all records via history server in batches, return combined XML.

    Returns:
        Combined XML string of all PubmedArticle elements.
    """
    all_articles_xml: list[str] = []

    for start in range(0, total, batch_size):
        params: dict[str, Any] = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": web_env,
            "retstart": start,
            "retmax": batch_size,
            "retmode": "xml",
            "rettype": "xml",
            "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        time.sleep(api_delay)
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()

        all_articles_xml.append(resp.text)
        logger.debug(
            "EFetch batch: retstart=%d, retmax=%d, got %d bytes",
            start,
            batch_size,
            len(resp.text),
        )

    # Combine into a single XML document
    if len(all_articles_xml) == 1:
        return all_articles_xml[0]

    # For multiple batches, extract PubmedArticle elements and wrap
    combined_articles: list[str] = []
    for xml_text in all_articles_xml:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            combined_articles.append(ET.tostring(article, encoding="unicode"))

    return (
        '<?xml version="1.0" ?>\n<PubmedArticleSet>\n'
        + "\n".join(combined_articles)
        + "\n</PubmedArticleSet>"
    )


def parse_pubmed_xml(
    xml_text: str,
    journal_abbreviation: str,
    journal_field: JournalField,
) -> list[ArticleRecord]:
    """Parse PubMed XML into ArticleRecord objects.

    Args:
        xml_text: EFetch XML response
        journal_abbreviation: NLM abbreviation for this query
        journal_field: Classification (general_medicine or medical_education)

    Returns:
        List of ArticleRecord (inclusion/exclusion not yet applied)
    """
    root = ET.fromstring(xml_text)
    records: list[ArticleRecord] = []

    for article_el in root.findall(".//PubmedArticle"):
        record = _parse_single_article(article_el, journal_abbreviation, journal_field)
        if record is not None:
            records.append(record)

    logger.info(
        "Parsed %d articles for %s (%s)",
        len(records),
        journal_abbreviation,
        journal_field.value,
    )
    return records


def _parse_single_article(
    article_el: ET.Element,
    journal_abbreviation: str,
    journal_field: JournalField,
) -> Optional[ArticleRecord]:
    """Parse a single <PubmedArticle> element."""
    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None

    # PMID
    pmid_el = medline.find("PMID")
    if pmid_el is None or not pmid_el.text:
        return None
    pmid = pmid_el.text

    # Article element
    article = medline.find("Article")
    if article is None:
        return None

    # Title
    title_el = article.find("ArticleTitle")
    title = _text_content(title_el) if title_el is not None else ""

    # Journal title (MedlineTA)
    journal_el = medline.find("MedlineJournalInfo/MedlineTA")
    journal = journal_el.text if journal_el is not None and journal_el.text else ""

    # DOI
    doi = _extract_doi(article_el)

    # PMCID
    pmcid = _extract_pmcid(article_el)

    # Publication date
    pub_date, pub_year = _extract_pub_date(article)

    # Publication types
    pub_types = _extract_publication_types(article)

    # Authors
    authors = _extract_authors(article)

    # Abstract
    abstract_el = article.find("Abstract")
    has_abstract = abstract_el is not None
    abstract_text = None
    if abstract_el is not None:
        abstract_parts = []
        for abs_text_el in abstract_el.findall("AbstractText"):
            label = abs_text_el.get("Label", "")
            text = _text_content(abs_text_el)
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract_text = " ".join(abstract_parts) if abstract_parts else None

    return ArticleRecord(
        pmid=pmid,
        doi=doi,
        title=title,
        journal=journal,
        journal_abbreviation=journal_abbreviation,
        journal_field=journal_field,
        pub_date=pub_date,
        pub_year=pub_year,
        publication_types=pub_types,
        authors=authors,
        has_abstract=has_abstract,
        abstract_text=abstract_text,
        pmcid=pmcid,
        has_pmc_fulltext=pmcid is not None,
    )


def efetch_by_pmids(
    pmids: list[str],
    email: str,
    api_key: str | None = None,
    batch_size: int = 200,
    api_delay: float = 0.34,
) -> str:
    """Fetch PubMed records by PMID list (no history server needed).

    Args:
        pmids: List of PubMed IDs.
        email: NCBI-required email.
        api_key: Optional NCBI API key (increases rate limit).
        batch_size: Max PMIDs per request (NCBI limit ~200 via GET).
        api_delay: Delay between requests in seconds.

    Returns:
        Combined XML string of all PubmedArticle elements.
    """
    all_articles_xml: list[str] = []

    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "xml",
            "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        time.sleep(api_delay)
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()

        all_articles_xml.append(resp.text)
        logger.info(
            "EFetch PMID batch %d-%d / %d (%d bytes)",
            start,
            min(start + batch_size, len(pmids)),
            len(pmids),
            len(resp.text),
        )

    if len(all_articles_xml) == 1:
        return all_articles_xml[0]

    combined_articles: list[str] = []
    for xml_text in all_articles_xml:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            combined_articles.append(ET.tostring(article, encoding="unicode"))

    return (
        '<?xml version="1.0" ?>\n<PubmedArticleSet>\n'
        + "\n".join(combined_articles)
        + "\n</PubmedArticleSet>"
    )


def extract_abstracts(xml_text: str) -> dict[str, str]:
    """Extract {pmid: abstract_text} from PubMed EFetch XML.

    Structured abstracts (with labels like BACKGROUND, METHODS, etc.)
    are formatted as "LABEL: text" and joined with spaces.
    """
    root = ET.fromstring(xml_text)
    result: dict[str, str] = {}

    for article_el in root.findall(".//PubmedArticle"):
        medline = article_el.find("MedlineCitation")
        if medline is None:
            continue
        pmid_el = medline.find("PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text

        article = medline.find("Article")
        if article is None:
            continue
        abstract_el = article.find("Abstract")
        if abstract_el is None:
            continue

        abstract_parts = []
        for abs_text_el in abstract_el.findall("AbstractText"):
            label = abs_text_el.get("Label", "")
            text = _text_content(abs_text_el)
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        if abstract_parts:
            result[pmid] = " ".join(abstract_parts)

    return result


def _text_content(el: ET.Element) -> str:
    """Get all text content including mixed content (text with sub-elements)."""
    return "".join(el.itertext()).strip()


def _extract_doi(article_el: ET.Element) -> str | None:
    """Extract DOI from PubmedData/ArticleIdList or ELocationID."""
    # Try ArticleIdList first
    pubmed_data = article_el.find("PubmedData")
    if pubmed_data is not None:
        for aid in pubmed_data.findall("ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                return aid.text.strip().lower()

    # Fallback: ELocationID
    article = article_el.find("MedlineCitation/Article")
    if article is not None:
        for eloc in article.findall("ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.text:
                return eloc.text.strip().lower()

    return None


def _extract_pmcid(article_el: ET.Element) -> str | None:
    """Extract PMCID from PubmedData/ArticleIdList."""
    pubmed_data = article_el.find("PubmedData")
    if pubmed_data is not None:
        for aid in pubmed_data.findall("ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pmc" and aid.text:
                pmcid = aid.text.strip()
                # Normalize: ensure "PMC" prefix
                if not pmcid.upper().startswith("PMC"):
                    pmcid = f"PMC{pmcid}"
                return pmcid
    return None


def _extract_pub_date(article: ET.Element) -> tuple[date | None, int | None]:
    """Extract publication date from Journal/JournalIssue/PubDate."""
    pub_date_el = article.find("Journal/JournalIssue/PubDate")
    if pub_date_el is None:
        return None, None

    year_el = pub_date_el.find("Year")
    month_el = pub_date_el.find("Month")
    day_el = pub_date_el.find("Day")

    year = int(year_el.text) if year_el is not None and year_el.text else None
    if year is None:
        # Try MedlineDate (e.g. "2025 Jan-Feb")
        medline_date_el = pub_date_el.find("MedlineDate")
        if medline_date_el is not None and medline_date_el.text:
            parts = medline_date_el.text.split()
            if parts and parts[0].isdigit():
                year = int(parts[0])

    if year is None:
        return None, None

    # Month mapping
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    month = None
    if month_el is not None and month_el.text:
        m = month_el.text.strip().lower()[:3]
        month = month_map.get(m)
        if month is None and m.isdigit():
            month = int(m)

    day = int(day_el.text) if day_el is not None and day_el.text else None

    pub_date_obj = None
    try:
        pub_date_obj = date(year, month or 1, day or 1)
    except (ValueError, TypeError):
        pass

    return pub_date_obj, year


def _extract_publication_types(article: ET.Element) -> list[str]:
    """Extract PublicationType list."""
    pub_types: list[str] = []
    for pt_el in article.findall("PublicationTypeList/PublicationType"):
        if pt_el.text:
            pub_types.append(pt_el.text.strip())
    return pub_types


def _extract_authors(article: ET.Element) -> list[str]:
    """Extract author names as 'LastName ForeName'."""
    authors: list[str] = []
    for author_el in article.findall("AuthorList/Author"):
        last = author_el.find("LastName")
        fore = author_el.find("ForeName")
        if last is not None and last.text:
            name = last.text
            if fore is not None and fore.text:
                name += f" {fore.text}"
            authors.append(name)
        else:
            # Collective name
            collective = author_el.find("CollectiveName")
            if collective is not None and collective.text:
                authors.append(collective.text)
    return authors
