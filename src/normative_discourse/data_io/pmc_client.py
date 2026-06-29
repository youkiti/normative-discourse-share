"""PMC full-text XML retrieval and JATS XML parsing.

Phase 2a: Fetch full-text XML from PMC via NCBI E-utilities (EFetch)
and parse JATS/NLM DTD structured XML into section-level plain text.

NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
JATS spec: https://jats.nlm.nih.gov/archiving/tag-library/
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from typing import Any


from normative_discourse.schemas import (
    PmcFetchManifest,
    PmcFetchResult,
    PmcFullText,
    PmcSection,
)

logger = logging.getLogger(__name__)

EFETCH_PMC_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_ARTICLE_URL_TEMPLATE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

# Tags whose content should be stripped from body text
_NOISE_TAGS = frozenset({
    "fig",
    "table-wrap",
    "ref-list",
    "fn-group",
    "supplementary-material",
    "ack",
    "glossary",
    "bio",
    "floats-group",
})

# sec-type attribute → normalized section label
_SEC_TYPE_MAP: dict[str, str] = {
    "intro": "introduction",
    "introduction": "introduction",
    "materials": "methods",
    "materials|methods": "methods",
    "methods": "methods",
    "subjects": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusions": "conclusion",
    "conclusion": "conclusion",
    "supplementary-material": "other",
}

# Title-text patterns → normalized section label (checked in order)
_TITLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bintroduction\b", re.I), "introduction"),
    (re.compile(r"\bbackground\b", re.I), "introduction"),
    (re.compile(r"\bproblem\b", re.I), "introduction"),
    (re.compile(r"\bcontext\b", re.I), "introduction"),
    (re.compile(r"\bmethod(?:s|ology)?\b", re.I), "methods"),
    (re.compile(r"\bmaterials?\s+(?:and|&)\s+methods?\b", re.I), "methods"),
    (re.compile(r"\bstudy\s+design\b", re.I), "methods"),
    (re.compile(r"\btrial\s+design\b", re.I), "methods"),
    (re.compile(r"\bparticipants?\b", re.I), "methods"),
    (re.compile(r"\bpatients?\b", re.I), "methods"),
    (re.compile(r"\bprocedures?\b", re.I), "methods"),
    (re.compile(r"\binterventions?\b", re.I), "methods"),
    (re.compile(r"\brandomi[sz](?:ation|ed)?(?:\s+and\s+masking)?\b", re.I), "methods"),
    (re.compile(r"\btrial\s+oversight\b", re.I), "methods"),
    (re.compile(r"\bstudy\s+oversight\b", re.I), "methods"),
    (re.compile(r"\bstudy\s+(?:population|cohort)\b", re.I), "methods"),
    (re.compile(r"\bpatient\s+population\b", re.I), "methods"),
    (re.compile(r"\bpopulation\s+descriptors?\b", re.I), "methods"),
    (re.compile(r"\bsample\s+size\b", re.I), "methods"),
    (re.compile(r"\bcovariates?\b", re.I), "methods"),
    (re.compile(r"\bdata\s+(?:collection|analysis|sources?)\b", re.I), "methods"),
    (re.compile(r"\bstatistical\s+analys(?:is|es)\b", re.I), "methods"),
    (re.compile(r"\bapproach\b", re.I), "methods"),
    (re.compile(r"\bdesign\b", re.I), "methods"),
    (re.compile(r"\bsetting\b", re.I), "methods"),
    (re.compile(r"\breflexivit(?:y|ies)\b", re.I), "methods"),
    (re.compile(r"\bsampling\s+and\s+recruitment\b", re.I), "methods"),
    (re.compile(r"\binterview\s+process\b", re.I), "methods"),
    (re.compile(r"\brandom(?:is|iz)(?:ation|ed)\b", re.I), "methods"),
    (re.compile(r"\bimplementation\b", re.I), "methods"),
    (re.compile(r"\bresults?\b", re.I), "results"),
    (re.compile(r"\bprincipal\s+findings?\b", re.I), "discussion"),
    (re.compile(r"\binterpretation\s+of\s+findings?\b", re.I), "discussion"),
    (re.compile(r"\bfindings?\b", re.I), "results"),
    (re.compile(r"\boutcomes?\b", re.I), "results"),
    (re.compile(r"\bsafety(?:\s+outcomes?)?\b", re.I), "results"),
    (re.compile(r"\befficacy\b", re.I), "results"),
    (re.compile(r"\bresponse\b", re.I), "results"),
    (re.compile(r"\bbaseline\s+characteristics?\b", re.I), "results"),
    (re.compile(r"\bpatient\s+characteristics?\b", re.I), "results"),
    (re.compile(r"\bevaluation\b", re.I), "results"),
    (re.compile(r"\bdiscussion\b", re.I), "discussion"),
    (re.compile(r"\bcomparison\s+with\s+(?:other\s+studies|prior\b)", re.I), "discussion"),
    (re.compile(r"\bcritical\s+reflection\b", re.I), "discussion"),
    (re.compile(r"\bpolicy\s+implications?\b", re.I), "discussion"),
    (re.compile(r"\bimplications?(?:\s+for\s+practice)?\b", re.I), "discussion"),
    (re.compile(r"\bpotential\s+for\s+clinical\s+application\b", re.I), "discussion"),
    (re.compile(r"\bnext\s+steps?\b", re.I), "discussion"),
    (re.compile(r"\brecommendations?\b", re.I), "discussion"),
    (re.compile(r"\blessons?\s+learned\b", re.I), "discussion"),
    (re.compile(r"\bapplicability\b", re.I), "discussion"),
    (re.compile(r"\bchallenges?\s+and\s+opportunit(?:y|ies)\b", re.I), "discussion"),
    (re.compile(r"\bfuture\s+directions?\b", re.I), "discussion"),
    (re.compile(r"\bconclusion(?:s)?\b", re.I), "conclusion"),
    (re.compile(r"\blimitation(?:s)?\b", re.I), "discussion"),
]

IMRAD_TYPES = frozenset({
    "abstract", "introduction", "methods", "results", "discussion", "conclusion",
})

_SKIP_SEC_TYPES = frozenset({
    "supplementary-material",
    "coi-statement",
    "data-availability",
})

_SKIP_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bauthor\s+contributions?\b", re.I),
    re.compile(r"\bconflicts?\s+of\s+interest\b", re.I),
    re.compile(r"\bconflict\s+of\s+interest\b", re.I),
    re.compile(r"\bethics\s+(?:statement|approval)\b", re.I),
    re.compile(r"\bdata\s+availability\b", re.I),
    re.compile(r"\bfunding\b", re.I),
    re.compile(r"\brole\s+of\s+the\s+funding\s+source\b", re.I),
    re.compile(r"\bprimary\s+funding\s+source\b", re.I),
    re.compile(r"\backnowledg(?:e)?ments?\b", re.I),
    re.compile(r"\bsupplementa(?:ry|l)\b", re.I),
    re.compile(r"\bcompeting\s+interests?\b", re.I),
    re.compile(r"\bother\s+disclosures?\b", re.I),
]

# Substring present in PMC XML when the publisher blocks full-text distribution
_PUBLISHER_RESTRICTION_NEEDLE = (
    "does not allow downloading of the full text in XML form"
)

# IMRaD boundary types used by introduction inference
_IMRAD_BOUNDARY_TYPES = frozenset({"methods", "results", "discussion", "conclusion"})

# Whitespace-tolerant "results" heading: PDF extraction sometimes splits the
# word ("RE SULTS", "RE S U LTS"), defeating the strict ``\bresults?\b`` pattern.
_RESULTS_HEADER_SPACED = re.compile(r"\br\s*e\s*s\s*u\s*l\s*t\s*s\b", re.I)
# Leading numbered-heading prefix, e.g. "3 | ", "3.1 ", "2.2.4) ".
_NUMBERED_HEADING_PREFIX = re.compile(r"^\s*\d+(?:\.\d+)*\s*[|.)\-:]?\s*")
_HTML_NOISE_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    ".fig",
    ".table-wrap",
    "section.tw",
    ".sm.xbox",
    ".supplementary-material",
    ".supplemental-material",
    ".supplementary-materials",
    ".associated-data",
    ".inline-equation",
    ".fm-sec",
    ".front-matter",
    ".history",
)
_HTML_SKIP_HEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bauthor\s+contributions?\b", re.I),
    re.compile(r"\bconflicts?\s+of\s+interest\b", re.I),
    re.compile(r"\bconflict\s+of\s+interest\b", re.I),
    re.compile(r"\bethics\s+(?:statement|approval)\b", re.I),
    re.compile(r"\bdata\s+availability\b", re.I),
    re.compile(r"\bfunding\b", re.I),
    re.compile(r"\backnowledg(?:e)?ments?\b", re.I),
    re.compile(r"\bsupplementa(?:ry|l)\b", re.I),
    re.compile(r"\bappendix\b", re.I),
    re.compile(r"\bcompeting\s+interests?\b", re.I),
]
_PMC_HTML_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhelpdesk\b", re.I),
    re.compile(r"\bsupport\s+center\b", re.I),
    re.compile(r"\binformation\s+from\s+the\s+box\s+above\b", re.I),
)


def infer_introduction_from_position(
    sections: list[PmcSection],
    min_words: int = 15,
    max_intro_words: int = 1800,
) -> list[PmcSection]:
    """Reclassify leading 'other' sections as 'introduction' by position.

    Many journals (NEJM, JAMA, Ann Intern Med) omit explicit "Introduction"
    headings.  Body text before the first methods/results/discussion heading
    is typically introductory content.  This function reclassifies such
    'other'-typed sections as 'introduction' when no explicit introduction
    section already exists.

    Args:
        sections: Ordered list of PmcSection objects.
        min_words: Minimum word count for reclassification.
        max_intro_words: Cumulative ceiling for the inferred introduction.
            The first qualifying section is always reclassified; further
            leading 'other' sections are reclassified only while the
            running total stays under this ceiling.  Prevents articles
            whose unmatched thematic headings span the whole body
            (Acad Med Innovation Reports) from collapsing entirely into
            "introduction".

    Returns:
        New list with leading 'other' sections reclassified (original untouched).
    """
    # Skip if introduction already exists
    if any(s.section_type == "introduction" for s in sections):
        return sections

    # Find index of the first IMRaD boundary section
    first_imrad_idx: int | None = None
    for i, s in enumerate(sections):
        if s.section_type in _IMRAD_BOUNDARY_TYPES:
            first_imrad_idx = i
            break

    result: list[PmcSection] = []
    inferred_words = 0
    inference_stopped = False
    for i, s in enumerate(sections):
        if s.section_type == "abstract":
            result.append(s)
        elif (
            s.section_type == "other"
            and (first_imrad_idx is None or i < first_imrad_idx)
            and s.word_count >= min_words
            and not inference_stopped
        ):
            if inferred_words > 0 and inferred_words + s.word_count > max_intro_words:
                # Ceiling reached: stop inferring so the introduction
                # stays contiguous; remaining sections keep their type.
                inference_stopped = True
                result.append(s)
                continue
            inferred_words += s.word_count
            result.append(PmcSection(
                section_type="introduction",
                section_title=s.section_title,
                text=s.text,
                word_count=s.word_count,
            ))
        else:
            result.append(s)
    return result


def infer_results_from_position(
    sections: list[PmcSection],
) -> list[PmcSection]:
    """Recover a Results section whose heading was mangled by PDF letter-spacing.

    Wiley/Med Educ PDFs use numbered IMRaD headings (``1 INTRODUCTION`` /
    ``2 METHODS`` / ``3 RESULTS`` / ``4 DISCUSSION``).  The PDF extractor
    sometimes splits the word "RESULTS" with stray spaces ("RE SULTS",
    "RE S U LTS"), so the strict ``\\bresults?\\b`` title pattern fails and the
    3-RESULTS block — plus its ``3.x`` subsections — is left typed ``other`` or
    inherits ``methods`` from the preceding section.  The article then has
    methods and discussion but an empty results column.

    This retypes the mangled results header and the sections that follow it (up
    to the first discussion) from ``other``/``methods`` to ``results`` when:

    1. the body has methods and discussion but NO results;
    2. the structural sections use a numbered heading scheme; and
    3. a section before the first discussion has a numbered title whose word
       matches a whitespace-tolerant "results" pattern.

    Conservative by design: only the contiguous block between the mangled
    results header and the first discussion section is retyped, and only
    sections currently typed ``other`` or ``methods``.  Returns a new list
    (original untouched); returns the input unchanged when nothing applies.
    """
    body = [s for s in sections if s.section_type != "abstract"]
    if not body:
        return sections
    types = {s.section_type for s in body}
    if "results" in types or "methods" not in types or "discussion" not in types:
        return sections
    # Require a numbered heading scheme on the structural sections so this only
    # touches Wiley-style numbered articles, not free-form innovation reports.
    if not any(
        _NUMBERED_HEADING_PREFIX.match(s.section_title or "")
        for s in body
        if s.section_type in ("methods", "discussion")
    ):
        return sections

    first_disc = next(
        (i for i, s in enumerate(sections) if s.section_type == "discussion"),
        None,
    )
    if first_disc is None:
        return sections

    res_start: int | None = None
    for i in range(first_disc):
        s = sections[i]
        if s.section_type == "abstract":
            continue
        title = s.section_title or ""
        if _NUMBERED_HEADING_PREFIX.match(title) and _RESULTS_HEADER_SPACED.search(title):
            res_start = i
            break
    if res_start is None:
        return sections

    result = list(sections)
    changed = False
    for i in range(res_start, first_disc):
        s = result[i]
        if s.section_type in ("other", "methods"):
            result[i] = PmcSection(
                section_type="results",
                section_title=s.section_title,
                text=s.text,
                word_count=s.word_count,
            )
            changed = True
    return result if changed else sections


def split_abstract_introduction(
    sections: list[PmcSection],
    pubmed_abstract: str,
    min_intro_words: int = 50,
) -> list[PmcSection]:
    """Split an oversized abstract section using PubMed abstract as boundary.

    Some journals (e.g. Ann Intern Med) have introduction text merged into the
    abstract section in PDFs.  This function uses the authoritative PubMed
    abstract to detect the true abstract boundary and moves trailing text into
    a new introduction section.

    Args:
        sections: Ordered list of PmcSection objects.
        pubmed_abstract: Full abstract text from PubMed API.
        min_intro_words: Minimum word count for the extracted introduction.

    Returns:
        New list with abstract split if applicable (original untouched).
    """
    if not pubmed_abstract or not pubmed_abstract.strip():
        return sections
    # Skip if introduction already exists
    if any(s.section_type == "introduction" for s in sections):
        return sections

    # Find abstract sections and concatenate their text
    abs_indices = [i for i, s in enumerate(sections) if s.section_type == "abstract"]
    if not abs_indices:
        return sections

    # Use last abstract section (most likely to contain overflow)
    abs_idx = abs_indices[-1]
    abs_section = sections[abs_idx]

    # Quick length check: parsed abstract should be notably longer than PubMed
    if abs_section.word_count < _word_count(pubmed_abstract) * 1.3:
        return sections

    # Find the last sentence of PubMed abstract in the parsed abstract text
    split_pos = _find_abstract_end(abs_section.text, pubmed_abstract)
    if split_pos is None:
        return sections

    # Split the text
    abstract_text = abs_section.text[:split_pos].strip()
    intro_text = abs_section.text[split_pos:].strip()

    if not intro_text or _word_count(intro_text) < min_intro_words:
        return sections

    # Build new sections list
    result: list[PmcSection] = []
    for i, s in enumerate(sections):
        if i == abs_idx:
            result.append(PmcSection(
                section_type="abstract",
                section_title=s.section_title,
                text=abstract_text,
                word_count=_word_count(abstract_text),
            ))
            result.append(PmcSection(
                section_type="introduction",
                section_title="",
                text=intro_text,
                word_count=_word_count(intro_text),
            ))
        else:
            result.append(s)
    return result


def anchor_body_start_with_pubmed(
    sections: list[PmcSection],
    pubmed_abstract: str,
    min_intro_words: int = 30,
    force_with_existing_intro: bool = False,
) -> list[PmcSection]:
    """Locate the abstract/body boundary using PubMed abstract as anchor.

    Searches across abstract AND leading 'other' sections (before the first
    IMRaD heading) to find where the PubMed abstract text ends.  Text after
    that boundary becomes a new introduction section.

    Compared to :func:`split_abstract_introduction`, this function:
    - Has **no word-count threshold** — always attempts matching.
    - Uses **case-insensitive** normalization.
    - Searches across 'other' sections in addition to abstract sections.
    - Falls back to sentence-level matching for structured abstracts.

    Args:
        sections: Ordered list of PmcSection objects.
        pubmed_abstract: Full abstract text from PubMed API.
        min_intro_words: Minimum word count for extracted introduction.

    Returns:
        New list with boundary corrected, or the original list unchanged.
    """
    if not pubmed_abstract or not pubmed_abstract.strip():
        return sections

    # Skip if introduction already exists (unless forced: oversized abstract
    # sections can swallow body text even when an introduction was parsed)
    if not force_with_existing_intro and any(
        s.section_type == "introduction" for s in sections
    ):
        return sections

    # Find the first IMRaD boundary section index
    first_imrad_idx: int | None = None
    for i, s in enumerate(sections):
        if s.section_type in _IMRAD_BOUNDARY_TYPES:
            first_imrad_idx = i
            break

    # Collect candidate sections: abstract + other before first IMRaD heading
    candidate_indices: list[int] = []
    for i, s in enumerate(sections):
        if first_imrad_idx is not None and i >= first_imrad_idx:
            break
        if s.section_type in ("abstract", "other"):
            candidate_indices.append(i)

    if not candidate_indices:
        return sections

    # Concatenate candidate text, tracking boundaries
    parts: list[str] = []
    for idx in candidate_indices:
        parts.append(sections[idx].text)
    concat_text = "\n\n".join(parts)

    # Find PubMed abstract end position in the concatenated text
    split_pos = _find_abstract_end(concat_text, pubmed_abstract)
    if split_pos is None:
        return sections

    # Split into abstract portion and introduction portion
    abstract_text = concat_text[:split_pos].strip()
    intro_text = concat_text[split_pos:].strip()

    if not intro_text or _word_count(intro_text) < min_intro_words:
        return sections

    # Build new sections list, replacing candidate sections
    result: list[PmcSection] = []
    candidate_set = frozenset(candidate_indices)
    first_placed = False

    for i, s in enumerate(sections):
        if i in candidate_set:
            if not first_placed:
                first_placed = True
                if abstract_text:
                    result.append(PmcSection(
                        section_type="abstract",
                        section_title=(
                            sections[candidate_indices[0]].section_title
                            or "Abstract"
                        ),
                        text=abstract_text,
                        word_count=_word_count(abstract_text),
                    ))
                result.append(PmcSection(
                    section_type="introduction",
                    section_title="",
                    text=intro_text,
                    word_count=_word_count(intro_text),
                ))
            # Skip remaining candidate sections (absorbed)
        else:
            result.append(s)

    return result


def _norm_ci(text: str) -> str:
    """Normalize text for comparison: collapse whitespace and lowercase.

    Also strips colons after word characters (e.g. "Background: text" →
    "background text") to handle structured abstract label differences
    between PubMed and PDF sources.
    """
    text = re.sub(r"(?<=\w):\s+", " ", text)  # "Background: text" → "Background text"
    return re.sub(r"\s+", " ", text).strip().lower()


def _split_sentences(text: str) -> list[str]:
    """Split normalized text into sentence-like segments.

    Returns only segments of 20+ characters to avoid matching on
    abbreviations or very short fragments.
    """
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def _locate_in_original(text: str, normalized_fragment: str) -> int | None:
    """Find where a normalized text fragment ends in the original text.

    Builds a regex from the normalized words, allowing optional colons and
    flexible whitespace between them, so that e.g. "background diabetes" will
    match "BACKGROUND: Diabetes" in the original.

    Returns:
        Character position in *text* just past the match, or None.
    """
    words = normalized_fragment.split()
    if not words:
        return None
    # Each word separated by optional colon + whitespace
    pattern = r"[:\s]+" .join(re.escape(w) for w in words)
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.end()
    return None


def _find_abstract_end(parsed_text: str, pubmed_abstract: str) -> int | None:
    """Find the position in parsed_text where the PubMed abstract ends.

    Uses a cascade of matching strategies:
    1. Suffix matching (last 120/80/50/30 chars) — fast, handles most cases
    2. Sentence-level matching — robust for structured abstracts where
       sub-heading labels differ between PubMed and PDF

    All comparisons are case-insensitive and tolerate colon/whitespace
    differences between PubMed and PDF sources.

    Returns:
        Character position in parsed_text just past the match, or None.
    """
    parsed_norm = _norm_ci(parsed_text)
    pubmed_norm = _norm_ci(pubmed_abstract)

    # Strategy 1: Suffix matching (case-insensitive, colon-normalized)
    for length in (120, 80, 50, 30):
        if len(pubmed_norm) < length:
            continue
        suffix = pubmed_norm[-length:]
        pos = parsed_norm.find(suffix)
        if pos >= 0:
            # Map back to original text via regex (handles colon/case diffs)
            return _locate_in_original(parsed_text, suffix)

    # Strategy 2: Sentence-level fallback
    # Walk PubMed sentences from last to first and return the position after
    # the last sentence that actually appears in the parsed text.  This
    # handles cases where PubMed has trailing metadata (e.g. trial
    # registration) that isn't in the PDF, or sub-heading reformatting.
    sentences = _split_sentences(pubmed_norm)
    for sent in reversed(sentences):
        pos = parsed_norm.find(sent)
        if pos >= 0:
            return _locate_in_original(parsed_text, sent)

    return None


def _map_norm_pos_to_original(original: str, norm_pos: int) -> int:
    """Map a position in normalized text back to the original text."""
    norm_idx = 0
    in_whitespace = False
    for i, ch in enumerate(original):
        if norm_idx >= norm_pos:
            return i
        if ch in (" ", "\t", "\n", "\r"):
            if not in_whitespace:
                norm_idx += 1
                in_whitespace = True
        else:
            norm_idx += 1
            in_whitespace = False
    return len(original)


# ---------------------------------------------------------------------------
# XML fetching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JATS XML parsing
# ---------------------------------------------------------------------------


def _extract_text(el: ET.Element) -> str:
    """Extract all text from an element, including inline sub-elements.

    Handles <italic>, <bold>, <xref>, <sup>, <sub>, etc. by extracting
    only their text content.
    """
    return "".join(el.itertext())


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _join_unique_texts(parts: list[str]) -> str | None:
    """Join normalized text parts while preserving order and dropping duplicates."""
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        cleaned = _normalize_whitespace(part)
        if cleaned and cleaned not in seen:
            kept.append(cleaned)
            seen.add(cleaned)
    return "\n\n".join(kept) if kept else None


def _extract_citation_tables_figures(article: ET.Element) -> str | None:
    """Extract references and table/figure captions outside the main body text."""
    groups: list[str] = []

    refs: list[str] = []
    for ref in article.findall(".//back//ref-list//ref") or article.findall(".//ref-list//ref"):
        text = _normalize_whitespace(_extract_text(ref))
        if text:
            refs.append(text)
    refs_text = _join_unique_texts(refs)
    if refs_text:
        groups.append("References\n" + refs_text)

    figures: list[str] = []
    for fig in article.findall(".//body//fig"):
        label_el = fig.find("label")
        caption_el = fig.find("caption")
        label = _normalize_whitespace(_extract_text(label_el)) if label_el is not None else ""
        caption = _normalize_whitespace(_extract_text(caption_el)) if caption_el is not None else ""
        text = ": ".join(part for part in (label, caption) if part)
        if text:
            figures.append(text)
    figures_text = _join_unique_texts(figures)
    if figures_text:
        groups.append("Figures\n" + figures_text)

    tables: list[str] = []
    for table_wrap in article.findall(".//body//table-wrap"):
        label_el = table_wrap.find("label")
        caption_el = table_wrap.find("caption")
        label = _normalize_whitespace(_extract_text(label_el)) if label_el is not None else ""
        caption = _normalize_whitespace(_extract_text(caption_el)) if caption_el is not None else ""
        text = ": ".join(part for part in (label, caption) if part)
        if text:
            tables.append(text)
    tables_text = _join_unique_texts(tables)
    if tables_text:
        groups.append("Tables\n" + tables_text)

    return "\n\n".join(groups) if groups else None


def _remove_noise_elements(body: ET.Element) -> None:
    """Remove noise elements (figures, tables, references, etc.) in place."""
    for tag in _NOISE_TAGS:
        for el in body.findall(f".//{tag}"):
            parent = _find_parent(body, el)
            if parent is not None:
                parent.remove(el)


def _find_parent(root: ET.Element, target: ET.Element) -> ET.Element | None:
    """Find the parent of target element within root."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


# Exact (whole-title) primary IMRaD headings that may pull a nested <sec>
# out of its parent section ("Conclusions" under Discussion).  Anything
# looser is a thematic sub-heading and inherits the parent: JATS nesting is
# authoritative for membership, so "Implementation and Analysis of the ..."
# under Results must stay results even though it contains "implementation".
_PRIMARY_CHILD_TITLE_MAP: dict[str, str] = {
    "introduction": "introduction",
    "background": "introduction",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "results": "results",
    "findings": "results",
    "results and discussion": "results",
    "discussion": "discussion",
    "limitations": "discussion",
    "strengths and limitations": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}


def classify_section_with_parent(
    sec_el: ET.Element,
    parent_imrad: str | None = None,
) -> str:
    """Classify a <sec> element with parent-context inheritance."""
    sec_type = (sec_el.get("sec-type") or "").strip().lower()
    if sec_type and sec_type in _SEC_TYPE_MAP:
        return _SEC_TYPE_MAP[sec_type]

    title_el = sec_el.find("title")
    title_text = (
        _extract_text(title_el).strip() if title_el is not None else ""
    )

    if parent_imrad in ("methods", "results", "discussion", "conclusion"):
        # JATS nesting is authoritative: a child <sec> belongs to its
        # parent section unless its own title is an explicit primary
        # IMRaD heading.  Loose pattern matches ("Outcomes" under
        # Methods, "Implementation and Analysis ..." under Results)
        # must not pull the child out of the running section.
        key = title_text.lower().rstrip(":. ")
        return _PRIMARY_CHILD_TITLE_MAP.get(key, parent_imrad)

    if title_text:
        for pattern, label in _TITLE_PATTERNS:
            if pattern.search(title_text):
                return label

    if parent_imrad and parent_imrad in IMRAD_TYPES and parent_imrad != "abstract":
        return parent_imrad

    return "other"


def _classify_section(sec_el: ET.Element) -> str:
    """Backward-compatible wrapper around parent-aware classification."""
    return classify_section_with_parent(sec_el, parent_imrad=None)


def _should_skip_section(sec_el: ET.Element) -> bool:
    """Determine if a section should be skipped as boilerplate/non-body."""
    sec_type = (sec_el.get("sec-type") or "").strip().lower()
    if sec_type in _SKIP_SEC_TYPES:
        return True

    title_el = sec_el.find("title")
    if title_el is not None:
        title_text = _extract_text(title_el).strip()
        for pattern in _SKIP_TITLE_PATTERNS:
            if pattern.search(title_text):
                return True
    return False


def _gather_paragraphs(el: ET.Element, out: list[str]) -> None:
    """Collect <p> prose at any depth below *el*, skipping nested <sec>.

    JATS wraps substantive body prose not only in direct <p> children but also
    inside <list>/<list-item> and <boxed-text> containers (e.g. Lancet results
    rendered as an ordered list of key findings).  We must descend into those
    so the prose is not silently dropped, while still leaving nested <sec>
    subtrees for separate top-level processing.
    """
    if el.tag == "sec":
        return
    if el.tag == "p":
        text = _normalize_whitespace(_extract_text(el))
        if text:
            out.append(text)
        return
    for child in el:
        _gather_paragraphs(child, out)


def _extract_section_text(sec_el: ET.Element) -> str:
    """Extract plain text from <p> elements within a <sec>, skipping nested <sec>.

    Paragraphs nested inside <list>/<boxed-text> containers are included; only
    nested <sec> subtrees are skipped (processed separately at top level).
    """
    paragraphs: list[str] = []

    for child in sec_el:
        _gather_paragraphs(child, paragraphs)

    return "\n\n".join(paragraphs)


def _word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def parse_pmc_xml(xml_text: str, pmid: str, pmcid: str) -> PmcFullText | None:
    """Parse PMC JATS XML into structured sections.

    Args:
        xml_text: Raw PMC XML string.
        pmid: PubMed ID.
        pmcid: PubMed Central ID.

    Returns:
        PmcFullText with parsed sections, or None on parse failure.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("XML parse error for %s: %s", pmcid, exc)
        return None

    # Find the <article> element (may be wrapped in <pmc-articleset>)
    article = root.find(".//article")
    if article is None:
        # Root might be <article> itself
        if root.tag == "article":
            article = root
        else:
            logger.warning("No <article> element found for %s", pmcid)
            return None

    sections: list[PmcSection] = []
    citation_tables_figures = _extract_citation_tables_figures(article)

    # --- Extract abstract ---
    abstract_el = article.find(".//front//abstract")
    if abstract_el is not None:
        abstract_parts: list[str] = []

        # Structured abstract with <sec> sub-sections
        abstract_secs = abstract_el.findall("sec")
        if abstract_secs:
            for abs_sec in abstract_secs:
                for p in abs_sec.findall("p"):
                    text = _normalize_whitespace(_extract_text(p))
                    if text:
                        abstract_parts.append(text)
        else:
            # Unstructured abstract — just <p> elements
            for p in abstract_el.findall("p"):
                text = _normalize_whitespace(_extract_text(p))
                if text:
                    abstract_parts.append(text)
            # Some abstracts have direct text without <p>
            if not abstract_parts:
                text = _normalize_whitespace(_extract_text(abstract_el))
                if text:
                    abstract_parts.append(text)

        if abstract_parts:
            abstract_text = "\n\n".join(abstract_parts)
            sections.append(PmcSection(
                section_type="abstract",
                section_title="Abstract",
                text=abstract_text,
                word_count=_word_count(abstract_text),
            ))

    # --- Extract body sections ---
    body = article.find(".//body")
    if body is not None:
        # Remove noise elements before parsing
        _remove_noise_elements(body)

        # Process top-level <sec> elements
        top_secs = body.findall("sec")
        if top_secs:
            # Capture <p> elements that appear before the first <sec>
            pre_sec_paragraphs: list[str] = []
            for child in body:
                if child.tag == "sec":
                    break
                if child.tag == "p":
                    text = _normalize_whitespace(_extract_text(child))
                    if text:
                        pre_sec_paragraphs.append(text)
            if pre_sec_paragraphs:
                pre_sec_text = "\n\n".join(pre_sec_paragraphs)
                wc = _word_count(pre_sec_text)
                if wc >= 15:
                    sections.append(PmcSection(
                        section_type="other",
                        section_title="",
                        text=pre_sec_text,
                        word_count=wc,
                    ))

            # Track the running IMRaD context across top-level siblings.
            # Some journals (Perspect Med Educ) emit results/discussion
            # subsections as flat top-level siblings rather than nested
            # children; an unmatched sibling following results/discussion
            # belongs to that running section, not "other".
            running_imrad: str | None = None
            for sec_el in top_secs:
                own_cls = classify_section_with_parent(sec_el, parent_imrad=None)
                if own_cls == "other" and running_imrad in ("results", "discussion"):
                    _process_sec_recursive(
                        sec_el, sections, parent_imrad=running_imrad
                    )
                else:
                    _process_sec_recursive(sec_el, sections, parent_imrad=None)
                    if own_cls in (
                        "introduction", "methods", "results",
                        "discussion", "conclusion",
                    ):
                        running_imrad = own_cls
        else:
            # No <sec> structure — extract all <p> from body
            body_parts: list[str] = []
            for p in body.findall("p"):
                text = _normalize_whitespace(_extract_text(p))
                if text:
                    body_parts.append(text)
            if body_parts:
                body_text = "\n\n".join(body_parts)
                sections.append(PmcSection(
                    section_type="other",
                    section_title="Body",
                    text=body_text,
                    word_count=_word_count(body_text),
                ))

    # Infer introduction from position if not explicitly present
    sections = infer_introduction_from_position(sections)

    total_wc = sum(s.word_count for s in sections)
    has_body = any(s.section_type != "abstract" for s in sections)

    return PmcFullText(
        pmid=pmid,
        pmcid=pmcid,
        sections=sections,
        citation_tables_figures=citation_tables_figures,
        total_word_count=total_wc,
        has_body=has_body,
    )


def _classify_html_heading(title: str) -> str:
    normalized = _normalize_whitespace(title)
    if not normalized:
        return "other"
    if re.search(r"\babstract\b", normalized, re.I):
        return "abstract"
    for pattern, label in _TITLE_PATTERNS:
        if pattern.search(normalized):
            return label
    return "other"


def _html_abstract_is_key_points(node) -> bool:
    """Detect a JAMA-style 'Key Points' summary box (Question/Findings/Meaning).

    These boxes are editorial summaries, not the article abstract, and must be
    dropped rather than scattered across results/conclusion sections.
    """
    heading = node.find(["h2", "h3"])
    if heading is not None:
        htext = _normalize_whitespace(heading.get_text(" ", strip=True))
        if re.search(r"\bkey\s*points?\b", htext, re.I):
            return True
    subs = {
        _normalize_whitespace(h.get_text(" ", strip=True)).lower()
        for h in node.find_all(["h3", "h4"])
    }
    return len(subs & {"question", "findings", "meaning"}) >= 2


def _extract_html_abstract(container) -> str | None:
    """Pull and remove PMC ``section.abstract`` blocks, returning their text.

    PMC HTML wraps the abstract (often structured with subheadings such as
    Importance/Objective/Results, or Purpose/Method/Conclusions) inside
    ``<section class="abstract">``.  The flat heading walk in
    :func:`parse_pmc_html` cannot capture these because the text lives under
    subheadings, leaving ``abstract`` empty and the content scattered into the
    body.  This collects each abstract block as a unit, drops Key Points boxes,
    and decomposes the nodes so the body walk does not re-process them.
    """
    parts: list[str] = []
    for node in container.select("section.abstract, div.abstract"):
        if _html_abstract_is_key_points(node):
            node.decompose()
            continue
        heading = node.find(["h2", "h3"])
        if (
            heading is not None
            and _classify_html_heading(heading.get_text(" ", strip=True)) == "abstract"
        ):
            heading.decompose()
        text = _normalize_whitespace(node.get_text(" ", strip=True))
        if text:
            parts.append(text)
        node.decompose()
    if not parts:
        return None
    return "\n\n".join(parts)


def _should_skip_html_heading(title: str) -> bool:
    normalized = _normalize_whitespace(title)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _HTML_SKIP_HEADING_PATTERNS)


def _collect_html_citation_text(soup: BeautifulSoup) -> str | None:
    references: list[str] = []
    for selector in (".ref-list li", ".ref-list .ref", "ol.ref-list li", "div.ref-list p"):
        for node in soup.select(selector):
            text = _normalize_whitespace(node.get_text(" ", strip=True))
            if text and text not in references:
                references.append(text)
    if not references:
        return None
    return "\n".join(references)


def _looks_like_pmc_error_html(html_text: str) -> bool:
    normalized = _normalize_whitespace(html_text)
    return all(pattern.search(normalized) for pattern in _PMC_HTML_ERROR_PATTERNS)


def parse_pmc_html(html_text: str, pmid: str, pmcid: str) -> PmcFullText | None:
    """Parse a PMC article HTML page into the shared full-text schema."""
    if _looks_like_pmc_error_html(html_text):
        return None
    soup = BeautifulSoup(html_text, "lxml")
    citation_tables_figures = _collect_html_citation_text(soup)
    for selector in _HTML_NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    for selector in (".ref-list", "ol.ref-list", "div.ref-list"):
        for node in soup.select(selector):
            node.decompose()

    container = None
    for selector in (".main-article-body", "main", "#mc", ".article-page", ".tsec", "body"):
        container = soup.select_one(selector)
        if container is not None:
            break
    if container is None:
        return None

    sections: list[PmcSection] = []
    abstract_text = _extract_html_abstract(container)
    if abstract_text:
        sections.append(PmcSection(
            section_type="abstract",
            section_title="Abstract",
            text=abstract_text,
            word_count=_word_count(abstract_text),
        ))
    current_title = ""
    current_type = "other"
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_type, current_parts
        text = "\n\n".join(part for part in current_parts if part)
        text = _normalize_whitespace(text)
        if text:
            sections.append(PmcSection(
                section_type=current_type,
                section_title=current_title,
                text=text,
                word_count=_word_count(text),
            ))
        current_title = ""
        current_type = "other"
        current_parts = []

    # Running h2 IMRaD context: PMC HTML is a two-level tree (h2 IMRaD
    # anchors, h3/h4 subsections) flattened by this walk.  Without parent
    # context, JAMA Results subsections ("Study Population", "Sensitivity
    # Analysis") re-classify independently and scatter into methods/other,
    # while the Methods subsection "Outcomes" hijacks results.  Mirror the
    # XML path: a subsection under a running IMRaD h2 inherits the parent
    # unless its own title is an explicit primary IMRaD heading.
    running_h2: str | None = None
    for node in container.find_all(["h2", "h3", "h4", "p", "li"], recursive=True):
        if node.name in {"h2", "h3", "h4"}:
            heading = _normalize_whitespace(node.get_text(" ", strip=True))
            if not heading:
                continue
            flush()
            current_title = heading
            classified = _classify_html_heading(heading)
            if _should_skip_html_heading(heading):
                current_type = "_skip"
                if node.name == "h2":
                    running_h2 = None
                continue
            if node.name == "h2":
                running_h2 = (
                    classified
                    if classified in (
                        "introduction", "methods", "results",
                        "discussion", "conclusion",
                    )
                    else None
                )
                current_type = classified
            elif running_h2 is not None:
                key = heading.lower().rstrip(":. ")
                current_type = _PRIMARY_CHILD_TITLE_MAP.get(key, running_h2)
            else:
                current_type = classified
            continue

        text = _normalize_whitespace(node.get_text(" ", strip=True))
        if not text:
            continue
        if current_type == "_skip":
            continue
        current_parts.append(text)

    flush()

    if not sections:
        paragraphs = [
            _normalize_whitespace(node.get_text(" ", strip=True))
            for node in container.find_all("p")
        ]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            return None
        body_text = "\n\n".join(paragraphs)
        sections = [PmcSection(
            section_type="other",
            section_title="Body",
            text=body_text,
            word_count=_word_count(body_text),
        )]

    sections = infer_introduction_from_position(sections)
    total_wc = sum(s.word_count for s in sections)
    has_body = any(s.section_type != "abstract" for s in sections)

    return PmcFullText(
        pmid=pmid,
        pmcid=pmcid,
        source_type="pmc_html",
        sections=sections,
        citation_tables_figures=citation_tables_figures,
        total_word_count=total_wc,
        has_body=has_body,
    )


def _process_sec_recursive(
    sec_el: ET.Element,
    sections: list[PmcSection],
    parent_imrad: str | None = None,
) -> None:
    """Process a <sec> element, recursing into nested <sec> children."""
    if _should_skip_section(sec_el):
        return

    section_type = classify_section_with_parent(sec_el, parent_imrad)

    title_el = sec_el.find("title")
    section_title = _normalize_whitespace(_extract_text(title_el)) if title_el is not None else ""

    text = _extract_section_text(sec_el)

    if text:
        sections.append(PmcSection(
            section_type=section_type,
            section_title=section_title,
            text=text,
            word_count=_word_count(text),
        ))

    for child_sec in sec_el.findall("sec"):
        _process_sec_recursive(child_sec, sections, parent_imrad=section_type)


# ---------------------------------------------------------------------------
# Batch fetch and save
# ---------------------------------------------------------------------------


_MANIFEST_FIELDNAMES = [
    "pmid",
    "pmcid",
    "journal",
    "status",
    "section_count",
    "word_count",
    "xml_save_path",
    "json_save_path",
    "html_save_path",
    "error_reason",
]


