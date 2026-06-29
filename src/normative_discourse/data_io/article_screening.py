"""Shared high-precision screening helpers for non-original articles."""

from __future__ import annotations

import re

from normative_discourse.schemas import ArticleRecord, JournalField

_EDUCATION_JOURNALS = {
    "JMIR Med Educ",
    "Acad Med",
    "Med Educ",
    "Med Teach",
    "Perspect Med Educ",
}

# NEJM DOI codes that represent original/primary research.
_NEJM_ALLOWED_DOI_CODES = {"oa", "sa"}

# JAMA non-Original-Investigation filter.
_JAMA_TITLE_EXCLUSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r":\s+A\s+Review\.?\s*$", re.IGNORECASE),
]

# Acad Med non-original filters.
_ACAD_MED_REVIEW_TITLE_PATTERN = re.compile(
    r"\b(?:scoping|systematic|narrative|rapid|umbrella)\s+review\b"
    r"|:\s+A\s+Review\.?\s*$",
    re.IGNORECASE,
)
_ACAD_MED_COMMENTARY_ABSTRACT_PATTERN = re.compile(
    r"\b(?:In\s+this|This)\s+(?:Invited\s+)?Commentary\b",
    re.IGNORECASE,
)
_ACAD_MED_PERSPECTIVE_ABSTRACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:In\s+this|This)\s+Scholarly\s+Perspective\b", re.IGNORECASE),
    re.compile(r"\bThis\s+perspective\s+offers\b", re.IGNORECASE),
]
_ACAD_MED_FOREWORD_TITLE_PATTERN = re.compile(r"\bForeword\b", re.IGNORECASE)

# Ann Intern Med non-original web columns and digests share the journal
# masthead but are not original research. Each lowercase DOI prefix under
# 10.7326/ is reserved for one column type, so DOI tests are precise.
# Original research uses the ``m`` prefix (e.g. m17-1234).
_AIM_NON_ORIGINAL_DOI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^10\.7326/g\d", re.IGNORECASE), "Annals Graphic Medicine"),
    (re.compile(r"^10\.7326/awho\d", re.IGNORECASE), "Annals for Hospitalists"),
    (re.compile(r"^10\.7326/awed\d", re.IGNORECASE), "Annals for Educators"),
    (re.compile(r"^10\.7326/w\d", re.IGNORECASE), "Web Exclusive column"),
    (re.compile(r"^10\.7326/a\d", re.IGNORECASE), "Annals On Call"),
    (re.compile(r"^10\.7326/(?:acpj|j)\d", re.IGNORECASE), "ACP Journal Club"),
    (re.compile(r"^10\.7326/aitc\d", re.IGNORECASE), "In the Clinic"),
    (re.compile(r"^10\.7326/istranslatedfrom", re.IGNORECASE), "translation duplicate"),
]

# BMJ News/columns/reviews filters.
# Context: PubMed publication_types for BMJ inherit the cited paper's type
# (e.g. RCT tag on NIHR Alert briefs), so DOI/type alone cannot discriminate
# originals from news digests. See memory feedback_bmj_news_exclusion.md.
_BMJ_NIHR_ABSTRACT_PATTERN = re.compile(
    r"evidence\.nihr\.ac\.uk|NIHR\s+Alert",
    re.IGNORECASE,
)
# NIHR briefs abstract frequently starts "The study<AuthorName>..." with the
# cited author running on with no space. The uppercase-letter anchor is the
# discriminator versus originals starting with "The study was ...".
_BMJ_NIHR_STUDY_PREFIX_PATTERN = re.compile(r"^\s*The\s+study[A-Z]")
_BMJ_COLUMN_TITLE_PATTERN = re.compile(
    r"^\s*(?:When\s+I\s+use\s+a\s+word\b|Minerva\b)",
    re.IGNORECASE,
)
_BMJ_NARRATIVE_REVIEW_ABSTRACT_PATTERN = re.compile(
    r"\b(?:In\s+)?[Tt]his\s+(?:article\s+we\s+|review\s+)(?:we\s+)?review(?:s)?\b",
)


# --- All-journal title-level review / meta-analysis safety net ---------------
# PubMed publication-type filtering (``exclude_publication_types``) misses
# secondary-evidence syntheses that are tagged only "Journal Article" (no
# "Meta-Analysis"/"Review" PT). The canonical example is PMID 34461040 (Lancet,
# "...: an individual participant-level data meta-analysis"), tagged
# "Journal Article; Research Support, Non-U.S. Gov't" only. This title-level
# screen runs for every journal as a backstop to the PT filter.
#
# Precision is essential: a naive "\breview\b"/"\bmeta-analysis\b" substring
# match produces ~80 false positives in this corpus (peer review, chart review,
# institutional review board, "comparison of guidelines", "reanalysis of
# network meta-analyses", "screening in systematic reviews and meta-analyses",
# etc. — all original research). So we match only:
#   1. unambiguous review-article types (scoping/narrative/umbrella/rapid/
#      integrative/literature review), which never appear in original-research
#      titles in this corpus; and
#   2. systematic-review / meta-analysis used as a *terminal article-type*
#      descriptor — an indefinite-article phrase at the very end of the title
#      ("...: a systematic review", "...: an updated meta-analysis",
#      "...: an individual participant data meta-analysis"). Tail-anchoring +
#      the indefinite article excludes titles that merely *mention* SR/MA as the
#      object of study.
_REVIEW_ARTICLE_TYPE_PATTERN = re.compile(
    r"\b(?:scoping|narrative|umbrella|rapid|integrative|literature)\s+reviews?\b",
    re.IGNORECASE,
)
_SYNTHESIS_TAIL_PATTERN = re.compile(
    r"[:–—-]\s+(?:an?|updated|pooled|prospective)\s+[\w\s,'-]*?"
    r"(?:systematic\s+reviews?(?:\s+and\s+meta[-\s]?analys[ie]s)?|meta[-\s]?analys[ie]s)"
    r"\.?\s*$",
    re.IGNORECASE,
)


def review_meta_analysis_title_reason(record: ArticleRecord) -> str | None:
    """Return an exclusion reason if the title marks a review/meta-analysis.

    All-journal backstop for secondary-evidence syntheses that slip through the
    publication-type filter (tagged only "Journal Article"). High precision by
    design — see the module note above.
    """
    title = record.title or ""
    if _REVIEW_ARTICLE_TYPE_PATTERN.search(title):
        return "Non-original (review article, title)"
    if _SYNTHESIS_TAIL_PATTERN.search(title):
        return "Non-original (systematic review / meta-analysis, title)"
    return None


def extract_nejm_doi_code(doi: str | None) -> str | None:
    """Extract the article-type code from a NEJM DOI."""
    if not doi:
        return None
    parts = doi.lower().split("/")
    if len(parts) < 2:
        return None
    suffix = parts[-1]
    if not suffix.startswith("nejm"):
        return None
    code = ""
    for ch in suffix[4:]:
        if ch.isalpha():
            code += ch
        else:
            break
    return code or None


def is_nejm_non_original(record: ArticleRecord) -> bool:
    """Return True if the record is a NEJM article with a non-original DOI code."""
    if not record.doi or "10.1056/nejm" not in record.doi.lower():
        return False
    code = extract_nejm_doi_code(record.doi)
    if code is None:
        return False
    return code not in _NEJM_ALLOWED_DOI_CODES


def jama_non_original_reason(record: ArticleRecord) -> str | None:
    """Return exclusion reason if the record is a non-original JAMA article."""
    if record.journal_abbreviation != "JAMA":
        return None
    if "10.1001/jama" not in (record.doi or "").lower():
        return None

    if not record.has_abstract:
        return "JAMA non-original (no abstract)"

    for pattern in _JAMA_TITLE_EXCLUSION_PATTERNS:
        if pattern.search(record.title):
            return f"JAMA non-original (title: {pattern.pattern})"

    return None


def acad_med_non_original_reason(record: ArticleRecord) -> str | None:
    """Return exclusion reason if the record is a non-original Acad Med article."""
    if record.journal_abbreviation != "Acad Med":
        return None

    title = record.title or ""
    abstract = record.abstract_text or ""

    if _ACAD_MED_REVIEW_TITLE_PATTERN.search(title):
        return "Acad Med non-original (review)"

    if _ACAD_MED_COMMENTARY_ABSTRACT_PATTERN.search(abstract):
        return "Acad Med non-original (commentary)"

    for pattern in _ACAD_MED_PERSPECTIVE_ABSTRACT_PATTERNS:
        if pattern.search(abstract):
            return "Acad Med non-original (scholarly perspective)"

    if _ACAD_MED_FOREWORD_TITLE_PATTERN.search(title):
        return "Acad Med non-original (foreword)"

    return None


def aim_non_original_reason(record: ArticleRecord) -> str | None:
    """Return exclusion reason if the record is a non-original AIM web column."""
    if record.journal_abbreviation != "Ann Intern Med":
        return None
    doi = (record.doi or "").strip()
    if not doi:
        return None
    for pattern, label in _AIM_NON_ORIGINAL_DOI_PATTERNS:
        if pattern.match(doi):
            return f"Ann Intern Med non-original ({label})"
    return None


def bmj_non_original_reason(record: ArticleRecord) -> str | None:
    """Return exclusion reason if the record is a non-original BMJ article.

    Targets three non-original categories that PubMed publication_types cannot
    reliably exclude:
    - NIHR Alert briefs (news digests of externally published studies)
    - "When I use a word" / Minerva columns
    - Narrative reviews written as commentary

    Relies on abstract/title phrasing because DOI prefix ``bmj.[letter]NNN``
    is shared with original research.
    """
    if record.journal_abbreviation != "BMJ":
        return None

    title = record.title or ""
    abstract = record.abstract_text or ""

    if _BMJ_COLUMN_TITLE_PATTERN.search(title):
        return "BMJ non-original (column)"

    if abstract:
        if _BMJ_NIHR_ABSTRACT_PATTERN.search(abstract):
            return "BMJ non-original (NIHR Alert brief)"

        if _BMJ_NIHR_STUDY_PREFIX_PATTERN.search(abstract):
            return "BMJ non-original (NIHR Alert brief, study-prefix)"

        if _BMJ_NARRATIVE_REVIEW_ABSTRACT_PATTERN.search(abstract):
            return "BMJ non-original (narrative review)"

    return None


def compile_journal_patterns(
    pattern_map: dict[str, list[str]] | None,
) -> dict[str, list[re.Pattern[str]]]:
    """Compile per-journal regex patterns."""
    if not pattern_map:
        return {}

    compiled_map: dict[str, list[re.Pattern[str]]] = {}
    for journal, patterns in pattern_map.items():
        if not patterns:
            continue
        compiled_map[journal.lower()] = [re.compile(pattern) for pattern in patterns]
    return compiled_map


def match_journal_patterns(
    journal: str,
    text: str,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> str | None:
    """Return matched pattern string if a journal-specific regex hits."""
    patterns = compiled_patterns.get(journal.lower())
    if not patterns or not text:
        return None
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def infer_non_original_reason(
    record: ArticleRecord,
    title_patterns: dict[str, list[re.Pattern[str]]] | None = None,
    abstract_patterns: dict[str, list[re.Pattern[str]]] | None = None,
) -> str | None:
    """Return a high-precision exclusion reason for non-original articles."""
    if jama_reason := jama_non_original_reason(record):
        return jama_reason

    if is_nejm_non_original(record):
        return f"NEJM non-original article type: {extract_nejm_doi_code(record.doi)}"

    if acad_med_reason := acad_med_non_original_reason(record):
        return acad_med_reason

    if aim_reason := aim_non_original_reason(record):
        return aim_reason

    if bmj_reason := bmj_non_original_reason(record):
        return bmj_reason

    if title_patterns:
        if matched_pattern := match_journal_patterns(
            record.journal_abbreviation,
            record.title,
            title_patterns,
        ):
            return f"Excluded title pattern: {matched_pattern}"

    if abstract_patterns:
        if matched_pattern := match_journal_patterns(
            record.journal_abbreviation,
            record.abstract_text or "",
            abstract_patterns,
        ):
            return f"Excluded abstract pattern: {matched_pattern}"

    # All-journal backstop: a review/meta-analysis tagged only "Journal Article"
    # that no journal-specific rule caught (e.g. PMID 34461040, Lancet). Runs
    # last so more-specific journal labels take precedence.
    if synthesis_reason := review_meta_analysis_title_reason(record):
        return synthesis_reason

    return None


def metadata_non_original_reason(
    *,
    journal_abbreviation: str,
    title: str,
    doi: str | None,
    has_abstract: bool,
    abstract_text: str | None,
    compiled_title_patterns: dict[str, list[re.Pattern[str]]] | None = None,
    compiled_abstract_patterns: dict[str, list[re.Pattern[str]]] | None = None,
) -> str | None:
    """Metadata-only wrapper used by corpus rebuilding on saved CSV rows."""
    journal_field = (
        JournalField.MEDICAL_EDUCATION
        if journal_abbreviation in _EDUCATION_JOURNALS
        else JournalField.GENERAL_MEDICINE
    )
    record = ArticleRecord(
        pmid="metadata-only",
        doi=doi,
        title=title,
        journal=journal_abbreviation,
        journal_abbreviation=journal_abbreviation,
        journal_field=journal_field,
        publication_types=[],
        has_abstract=has_abstract,
        abstract_text=abstract_text,
    )
    return infer_non_original_reason(
        record,
        title_patterns=compiled_title_patterns,
        abstract_patterns=compiled_abstract_patterns,
    )
