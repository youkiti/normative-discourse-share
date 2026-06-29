"""Pydantic data models for normative-discourse pipeline."""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class JournalField(str, Enum):
    """Journal discipline classification."""

    GENERAL_MEDICINE = "general_medicine"
    MEDICAL_EDUCATION = "medical_education"


class ArticleRecord(BaseModel):
    """Metadata for a single PubMed article."""

    pmid: str = Field(description="PubMed ID")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    title: str = Field(description="Article title")
    journal: str = Field(description="Journal title (MedlineTA)")
    journal_abbreviation: str = Field(
        description="NLM title abbreviation used in PubMed query"
    )
    journal_field: JournalField = Field(
        description="general_medicine or medical_education"
    )
    pub_date: Optional[date] = Field(
        default=None, description="Publication date (YYYY-MM-DD)"
    )
    pub_year: Optional[int] = Field(default=None, description="Publication year")
    publication_types: list[str] = Field(
        default_factory=list,
        description="MeSH publication types assigned by PubMed",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names (LastName ForeName)",
    )
    has_abstract: bool = Field(default=False, description="Whether abstract is present")
    abstract_text: Optional[str] = Field(
        default=None, description="Abstract text (for reference)"
    )

    # PMC availability (for PMC XML-first strategy)
    pmcid: Optional[str] = Field(
        default=None, description="PubMed Central ID (e.g. PMC1234567)"
    )
    has_pmc_fulltext: bool = Field(
        default=False, description="Whether full-text XML is available in PMC"
    )

    # Inclusion/exclusion status (Stage 2 filter results)
    included: bool = Field(
        default=True, description="Whether article passed Stage 2 filter"
    )
    exclusion_reason: Optional[str] = Field(
        default=None, description="Reason for exclusion (None if included)"
    )


class CollectionStats(BaseModel):
    """Statistics for a single journal's collection run."""

    journal_abbreviation: str
    journal_field: JournalField
    query_string: str
    total_retrieved: int = 0
    stage2_excluded: int = 0
    stage2_included: int = 0
    pmc_available: int = 0
    pmc_unavailable: int = 0
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)


class CollectionLog(BaseModel):
    """Overall collection log for a Phase 1 run."""

    run_date: str
    date_range: str
    total_journals: int = 0
    journal_stats: list[CollectionStats] = Field(default_factory=list)
    total_retrieved: int = 0
    total_included: int = 0
    total_excluded: int = 0


class PRISMACounts(BaseModel):
    """Counts for PRISMA flow diagram."""

    records_identified: int = Field(
        description="Total records from PubMed search (Stage 1)"
    )
    records_after_dedup: int = Field(
        description="Records after deduplication (should equal identified for PubMed)"
    )
    records_screened: int = Field(
        description="Records screened at Stage 2"
    )
    records_excluded: int = Field(
        description="Records excluded at Stage 2"
    )
    exclusion_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Exclusion counts by reason",
    )
    articles_included: int = Field(
        description="Final articles included for analysis"
    )
    by_field: dict[str, int] = Field(
        default_factory=dict,
        description="Included counts by journal field",
    )
    pmc_available: int = Field(
        default=0,
        description="Included articles with PMC full-text XML available",
    )
    pmc_unavailable: int = Field(
        default=0,
        description="Included articles without PMC full-text (need PDF pipeline)",
    )


# --- Phase 2b: Full-text retrieval schemas ---


class ContentFormat(str, Enum):
    """Format of retrieved full-text content."""

    PDF = "pdf"
    XML = "xml"
    PLAIN_TEXT = "text"


class RetrievalResult(BaseModel):
    """Unified result model for full-text retrieval."""

    pmid: str = Field(description="PubMed ID of the article")
    doi: Optional[str] = Field(default=None, description="DOI of the article")
    journal: str = Field(default="", description="Journal abbreviation")
    status: str = Field(
        description=(
            "Retrieval status: "
            "'retrieved' | 'not_found' | 'invalid' | 'error' | 'manual_needed'"
        )
    )
    tier: Optional[str] = Field(
        default=None,
        description="Which tier succeeded: 'tier1_oa' | 'tier2_publisher' | None",
    )
    content_format: Optional[ContentFormat] = Field(
        default=None, description="Format of retrieved content"
    )
    source: Optional[str] = Field(
        default=None,
        description=(
            "Source that provided the content, e.g. "
            "'unpaywall', 'elsevier_api', 'crossref_tdm'"
        ),
    )
    save_path: Optional[Path] = Field(
        default=None, description="Path where the file was saved"
    )
    error_reason: Optional[str] = Field(
        default=None, description="Reason for failure (when status != 'retrieved')"
    )


class RetrievalManifest(BaseModel):
    """Summary of a full-text retrieval run."""

    run_date: str = Field(description="ISO-format run timestamp")
    total_articles: int = Field(default=0)
    retrieved: int = Field(default=0)
    not_found: int = Field(default=0)
    errors: int = Field(default=0)
    manual_needed: int = Field(default=0)
    by_journal: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-journal counts: {journal: {retrieved, not_found, ...}}",
    )
    by_tier: dict[str, int] = Field(
        default_factory=dict,
        description="Counts by successful tier: {tier1_oa: N, tier2_publisher: N}",
    )
    by_format: dict[str, int] = Field(
        default_factory=dict,
        description="Counts by content format: {pdf: N, xml: N, text: N}",
    )


# --- Phase 2a: PMC full-text XML retrieval schemas ---


class PmcSection(BaseModel):
    """A single section extracted from PMC JATS XML."""

    section_type: str = Field(
        description=(
            "Normalized section label: "
            "'abstract' | 'introduction' | 'methods' | 'results' | "
            "'discussion' | 'conclusion' | 'other'"
        )
    )
    section_title: str = Field(
        default="", description="Original section title from XML"
    )
    text: str = Field(description="Extracted plain text of the section")
    word_count: int = Field(default=0, description="Word count of extracted text")


class PmcFullText(BaseModel):
    """Parsed full-text content from a PMC JATS XML article."""

    pmid: str = Field(description="PubMed ID")
    pmcid: str = Field(description="PubMed Central ID (e.g. PMC1234567)")
    source_type: str = Field(
        default="pmc_xml", description="Always 'pmc_xml' for this schema"
    )
    sections: list[PmcSection] = Field(
        default_factory=list, description="Ordered list of parsed sections"
    )
    citation_tables_figures: Optional[str] = Field(
        default=None,
        description="References and table/figure caption text extracted separately",
    )
    total_word_count: int = Field(
        default=0, description="Total word count across all sections"
    )
    has_body: bool = Field(
        default=True,
        description="True if non-abstract body sections exist",
    )


class PmcFetchResult(BaseModel):
    """Result of fetching a single PMC full-text XML."""

    pmid: str = Field(description="PubMed ID")
    pmcid: str = Field(description="PubMed Central ID")
    status: str = Field(
        description=(
            "Fetch status: "
            "'fetched' | 'parsed' | 'abstract_only' | 'not_in_pmc' | 'error'"
        )
    )
    xml_save_path: Optional[Path] = Field(
        default=None, description="Path to saved raw XML"
    )
    json_save_path: Optional[Path] = Field(
        default=None, description="Path to saved parsed JSON"
    )
    html_save_path: Optional[Path] = Field(
        default=None, description="Path to saved raw HTML fallback page"
    )
    section_count: int = Field(default=0, description="Number of sections parsed")
    word_count: int = Field(default=0, description="Total word count")
    error_reason: Optional[str] = Field(
        default=None, description="Error details if status is 'error'"
    )


class PmcFetchManifest(BaseModel):
    """Summary of a PMC XML fetch run (Phase 2a)."""

    run_date: str = Field(description="ISO-format run timestamp")
    total_articles: int = Field(default=0)
    fetched: int = Field(default=0, description="XML fetched and saved")
    parsed: int = Field(
        default=0, description="XML fetched, saved, and parsed with body"
    )
    abstract_only: int = Field(
        default=0,
        description="XML fetched but body absent (publisher restriction)",
    )
    not_in_pmc: int = Field(default=0)
    errors: int = Field(default=0)
    by_journal: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-journal counts: {journal: {fetched, parsed, ...}}",
    )


# --- Phase 3: Adobe PDF Extract schemas ---


class ExtractionResult(BaseModel):
    """Result of extracting structured content from a single PDF."""

    pmid: str = Field(description="PubMed ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    journal: str = Field(default="", description="Journal abbreviation")
    status: str = Field(
        description=(
            "Extraction status: "
            "'extracted' | 'error' | 'quota_exhausted' | 'skipped'"
        )
    )
    pdf_path: Optional[Path] = Field(
        default=None, description="Path to source PDF"
    )
    json_path: Optional[Path] = Field(
        default=None, description="Path to structuredData.json output"
    )
    output_dir: Optional[Path] = Field(
        default=None, description="Path to extraction output directory"
    )
    table_count: int = Field(default=0, description="Number of tables extracted")
    figure_count: int = Field(default=0, description="Number of figures extracted")
    page_count: int = Field(default=0, description="Estimated number of pages in PDF")
    estimated_transactions: int = Field(
        default=0,
        description="Estimated Adobe API transactions consumed (1 per 5 pages)",
    )
    error_reason: Optional[str] = Field(
        default=None, description="Error details if status != 'extracted'"
    )


class ExtractionManifest(BaseModel):
    """Summary of a PDF extraction run (Phase 3)."""

    run_date: str = Field(description="ISO-format run timestamp")
    total_pdfs: int = Field(default=0)
    extracted: int = Field(default=0)
    errors: int = Field(default=0)
    skipped: int = Field(default=0, description="Already extracted (resume)")
    total_tables: int = Field(default=0)
    total_figures: int = Field(default=0)
    total_transactions_used: int = Field(default=0)
    by_journal: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-journal counts: {journal: {extracted, errors, ...}}",
    )
    credential_index_used: int = Field(
        default=0,
        description="Index of last credential set used (0-based)",
    )


# --- PMC XML to Markdown conversion schemas ---


class MarkdownConversionResult(BaseModel):
    """Result of converting a single PMC XML to normalized Markdown."""

    pmid: str = Field(description="PubMed ID")
    pmcid: str = Field(default="", description="PubMed Central ID")
    journal: str = Field(default="", description="Journal abbreviation")
    status: str = Field(
        description=(
            "Conversion status: "
            "'converted' | 'abstract_only' | 'error'"
        )
    )
    md_save_path: Optional[Path] = Field(
        default=None, description="Path to output .md file"
    )
    section_count: int = Field(default=0, description="Number of sections in output")
    word_count: int = Field(default=0, description="Total word count in output")
    imrad_coverage: dict[str, int] = Field(
        default_factory=dict,
        description="Word count per IMRaD section type",
    )
    has_body: bool = Field(default=True, description="Whether XML had a <body> element")
    error_reason: Optional[str] = Field(
        default=None, description="Error details if status is 'error'"
    )


# --- Phase 4: Analysis corpus ---


class CorpusRecord(BaseModel):
    """A single row in the analysis corpus Parquet file.

    Combines article metadata with IMRaD section text columns.
    """

    pmid: str = Field(description="PubMed ID")
    doi: Optional[str] = Field(default=None)
    title: str = Field(default="")
    journal: str = Field(default="")
    journal_field: str = Field(default="")
    pub_year: Optional[int] = Field(default=None)
    pub_date: Optional[str] = Field(default=None)
    authors: str = Field(default="", description="Semicolon-separated author list")
    n_authors: int = Field(default=0)
    publication_types: str = Field(default="", description="Semicolon-separated")
    pmcid: Optional[str] = Field(default=None)
    source_format: Optional[str] = Field(
        default=None, description="pmc_xml | pdf | None"
    )

    # Abstract
    abstract_pubmed: Optional[str] = Field(
        default=None, description="Abstract fetched from PubMed E-utilities"
    )

    # IMRaD section text columns (from PDF/PMC parser)
    abstract: Optional[str] = Field(default=None)
    introduction: Optional[str] = Field(default=None)
    methods: Optional[str] = Field(default=None)
    results: Optional[str] = Field(default=None)
    discussion: Optional[str] = Field(default=None)
    conclusion: Optional[str] = Field(default=None)
    other_sections: Optional[str] = Field(default=None)
    body_full: Optional[str] = Field(
        default=None, description="All sections except abstract, concatenated"
    )
    citation_tables_figures: Optional[str] = Field(
        default=None,
        description="References and table/figure caption text extracted separately",
    )
    is_non_imrad: bool = Field(default=False)
    non_imrad_reason: Optional[str] = Field(default=None)
    has_body: bool = Field(default=False)
    word_count: int = Field(default=0)


# --- Phase 5: Deontic expression detection ---

# Expression groups for statistical analysis (5 groups; see doc/protocol_memo.md §5.1)
EXPRESSION_GROUPS = (
    "must",
    "should",
    "need_to",
    "have_to",
    "required_to",
)


class SectionInventoryRecord(BaseModel):
    """One row per article x section in section_inventory.parquet.

    Stores normalised word count and per-group deontic expression tallies
    so that the main analysis never depends on re-running detection.
    """

    pmid: str = Field(description="PubMed ID")
    section: str = Field(description="e.g. 'abstract_pubmed', 'introduction'")
    word_count: int = Field(description="Word count of normalised text for this section")
    total_deontic: int = Field(default=0, description="Affirmative + negation total")
    affirmative_count: int = Field(default=0)
    negation_count: int = Field(default=0)
    # Per-group counts (affirmative)
    must: int = Field(default=0)
    should: int = Field(default=0)
    need_to: int = Field(default=0)
    have_to: int = Field(default=0)
    required_to: int = Field(default=0)
    # Per-group counts (negation) — all five groups carry a negated variant
    must_neg: int = Field(default=0)
    should_neg: int = Field(default=0)
    need_to_neg: int = Field(default=0)
    have_to_neg: int = Field(default=0)
    required_to_neg: int = Field(default=0)
    rate_per_1000: float = Field(
        default=0.0, description="total_deontic / word_count * 1000"
    )


class MatchRecord(BaseModel):
    """One row per detected deontic expression in matches.parquet."""

    pmid: str = Field(description="PubMed ID")
    section: str = Field(description="Section where match was found")
    expression: str = Field(description="Canonical surface form, e.g. 'should not'")
    expression_group: str = Field(description="Lemma group, e.g. 'should'")
    is_negation: bool = Field(default=False)
    start_char: int = Field(description="Character offset in normalised text")
    end_char: int = Field(description="Character offset end")
    context_before: str = Field(default="", description="N words before match")
    context_after: str = Field(default="", description="N words after match")


# ---------------------------------------------------------------------------
# Phase 7: Full-text validation review
# ---------------------------------------------------------------------------

REVIEW_FIELD_OPTIONS: dict[str, list[str]] = {
    "review_actual_style": ["imrad", "non_imrad", "unclear"],
    "review_label_ok": ["yes", "no", "maybe"],
    "review_boundary_ok": ["yes", "no", "maybe"],
    "review_noise_ok": ["yes", "no", "minor"],
    "review_methods_results_missing_true": ["yes", "no", "n/a"],
    "review_severity": ["none", "minor", "major"],
}


class FulltextReviewResult(BaseModel):
    """Single article's review assessment (used by both human and Gemini)."""

    pmid: str = Field(description="PubMed ID")
    review_actual_style: str = Field(
        default="", description="imrad | non_imrad | unclear"
    )
    review_label_ok: str = Field(default="", description="yes | no | maybe")
    review_boundary_ok: str = Field(default="", description="yes | no | maybe")
    review_noise_ok: str = Field(default="", description="yes | no | minor")
    review_methods_results_missing_true: str = Field(
        default="", description="yes | no | n/a"
    )
    review_severity: str = Field(default="", description="none | minor | major")
    review_error_causes: str = Field(
        default="",
        description="Pipe-separated error causes, e.g. 'boundary_misplacement|noise_metadata'",
    )
    review_notes: str = Field(default="")
    status: str = Field(default="ok", description="ok | error")
    error_detail: str = Field(default="")
