"""Build the analysis corpus Parquet from metadata CSV + cleaned-text JSONs.

Merges article metadata with IMRaD section-level text and outputs a single
Parquet file with ZSTD compression.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from normative_discourse.config.settings import get_pubmed_config, get_remediation_config
from normative_discourse.data_io.article_screening import (
    compile_journal_patterns,
    metadata_non_original_reason,
)
from normative_discourse.data_io.pmc_client import (
    _TITLE_PATTERNS,
    anchor_body_start_with_pubmed,
    infer_introduction_from_position,
    infer_results_from_position,
    split_abstract_introduction,
)
from normative_discourse.schemas import PmcFullText, PmcSection

logger = logging.getLogger(__name__)

# IMRaD section types that get their own column
IMRAD_COLUMNS = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
)

# JAMA Special Communication PMIDs excluded manually (Phase 2b)
_JAMA_SPECIAL_COMMUNICATION_PMIDS = frozenset(
    {"41182880", "40899949", "40810984", "40009364", "39761025", "39425948"}
)

# Conference-abstract supplements that slipped past screening but are not
# original articles.  34227942 is a Lancet 2021 Public Health Science meeting
# abstract: its only "body" is the affiliation block and its parsed abstract is
# bled in from a *different* abstract in the same supplement (title = Gaza pain
# control, abstract text = COVID-19 elective surgery).  The 16 sibling
# 34227xxx abstracts were already screened out; this one was kept only because
# the pmc_xml parse produced a spurious body.  (2026-06-11 eyeball follow-up.)
#
# NOTE: bulk Lancet meeting-supplement exclusions live at the source-CSV layer
# (scripts/apply_lancet_supplement_filter.py) so they are PRISMA-counted; this
# frozenset is only the late straggler that the CSV pass missed.
_CONFERENCE_ABSTRACT_PMIDS = frozenset({"34227942"})

# Articles whose Adobe PDF extract captured the wrong content (appendix /
# survey instrument / consent form instead of the article prose), so the
# stored "body" is not the article.  No parser change can recover these — the
# extraction layer captured the wrong pages.  31567172 ("Open Office Space…")
# extracted the Mission Hall Workplace Research Survey questionnaire as its
# body.  (2026-06-11 eyeball follow-up, acadmed_pdf_oversized2 Recommendation 4.)
# 31076174 (Lancet "Global alcohol exposure 1990-2017... modelling study", a GBD
# paper) extracted the body of an *unrelated* document instead: a Candida
# albicans fungal-biofilm thesis (intro = "Fungal pathogens...", a Malay-language
# abstract block, and a units glossary).  No parser change can recover the real
# article -- the wrong PDF was downloaded.  This is a genuine corpus member, so
# it is also logged in reports/redownload_wrong_pdf_2026-06-15.txt for re-fetch;
# until the correct PDF is retrieved it must be excluded.  (2026-06-15 IMRaD
# residual-63 cleanup.)
_BAD_EXTRACT_PMIDS = frozenset({"31567172", "31076174"})

# JAMA "Clinical Evidence Synopsis" — a short, structured summary of an existing
# systematic review (Cochrane synopsis format), not original primary research.
# 29710147 ("Hydrolyzed Formula Compared With Standard Formula for Preterm
# Infants") is a ~520-word synopsis that slipped past PubMed publication-type and
# word-count screening.  It surfaced only after the 2026-06-27 sanitisation fix
# restored its body above the 500-word minimum; excluded here on article-type
# grounds, consistent with the non-original-article eligibility rule.
_CLINICAL_EVIDENCE_SYNOPSIS_PMIDS = frozenset({"29710147"})

# Med Teach article types considered original research.
# Must stay in sync with scripts/apply_stage3_2016_2023.py:
# "Original Articles" is the pre-2023 T&F section label and
# "Research Report(s)" the pre-2017 one.
_MED_TEACH_ORIGINAL_TYPES = frozenset(
    {
        "articles",
        "article",
        "research article",
        "research articles",
        "original article",
        "original articles",
        "research report",
        "research reports",
    }
)

# Analysis-time minimum length. Short reports below this threshold are excluded
# from the final analysis corpus per the 2026-03-07 QA decision.
SHORT_ARTICLE_THRESHOLD = 500

_AUTHOR_DEGREE_PATTERN = re.compile(
    r"\b(?:MD|PhD|ScD|MBBS|MPH|MSc|MS|MA|BA|MEd|MHPE|EdD|DO|DPhil|DrPH|"
    r"MBChB|BSc|BMedSci|RN|FACP|FRCP|FRCPC)\b",
    re.IGNORECASE,
)
_CONTENT_CUE_PATTERN = re.compile(
    r"\b(?:background|importance|objective|objectives|purpose|methods?|"
    r"results?|conclusions?|discussion|study|trial|patients?|participants?)\b",
    re.IGNORECASE,
)
_CAPTION_LABEL_PATTERN = re.compile(
    r"^\s*(?:figure|fig\.?|table)\s*[a-z0-9]+(?:[\s.:)\-]|$)",
    re.IGNORECASE,
)
# A body sentence that *refers* to a figure/table ("Figure 1A shows trends...",
# "Table 2 presents...", "Figure S8 in the Supplementary Appendix shows...")
# also matches _CAPTION_LABEL_PATTERN but is genuine prose, not a caption.
# Discriminator: after the label (and any "and/to/-" list of further numbers)
# the next word is lowercase (a referring verb), whereas a real caption title
# begins with a period/colon or a Title-Case noun phrase.
_FIGURE_REFERENCE_PROSE_PATTERN = re.compile(
    r"^\s*(?:figures?|fig\.?|tables?)\s*[A-Za-z]?\d+[A-Za-z]?"
    r"(?:\s*(?:and|to|through|&|[,–-]+)\s*"
    r"(?:figures?|tables?\s*)?[A-Za-z]?\d+[A-Za-z]?)*"
    r"\s+([A-Za-z]+)",
    re.IGNORECASE,
)


def _looks_like_figure_reference_prose(text: str) -> bool:
    """True when a "Figure/Table N ..." paragraph is a referring sentence.

    Distinguishes body prose ("Figure 1A shows...") from a figure/table caption
    ("Figure 1. Age-specific incidence...") so the latter alone is dropped as
    noise.  Returns True when the word following the figure/table reference is
    lowercase (typically a verb such as shows/presents/summarizes).
    """
    m = _FIGURE_REFERENCE_PROSE_PATTERN.match(text)
    return bool(m and m.group(1)[:1].islower())
_ANALYSIS_METADATA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*doi\s*[:(]", re.IGNORECASE),
    re.compile(r"^\s*(?:accepted|published)\s+(?:for|online)\b", re.IGNORECASE),
    re.compile(r"^\s*author\s+(?:affiliations?|information|contributions?)\b", re.IGNORECASE),
    re.compile(r"^\s*affiliations?\s*:", re.IGNORECASE),
    re.compile(r"^\s*correspond(?:ence|ing\s+author)\b", re.IGNORECASE),
    re.compile(r"^\s*article\s+information\b", re.IGNORECASE),
    re.compile(r"^\s*copyright\b", re.IGNORECASE),
    re.compile(r"^\s*this\s+article\s+was\s+", re.IGNORECASE),
    re.compile(r"^\s*downloaded\s+from\b", re.IGNORECASE),
    re.compile(r"^\s*(?:funding|funding/support)\s*:", re.IGNORECASE),
    re.compile(r"^\s*supported\s+by\b", re.IGNORECASE),
    re.compile(r"^\s*other disclosures?\s*:", re.IGNORECASE),
    re.compile(r"^\s*disclosures?\s*:", re.IGNORECASE),
    re.compile(r"^\s*ethical approval\s*:", re.IGNORECASE),
    re.compile(r"^\s*disclaimers?\s*:", re.IGNORECASE),
    re.compile(r"^\s*previous posting\s*:", re.IGNORECASE),
    re.compile(r"^\s*acknowledg(?:e)?ments?\s*:", re.IGNORECASE),
    re.compile(r"^\s*disclosure forms provided by the authors\b", re.IGNORECASE),
    re.compile(r"^\s*data sharing statement provided by the authors\b", re.IGNORECASE),
    re.compile(r"^\s*author contributions? (?:are|is) available\b", re.IGNORECASE),
)
_REFERENCE_ENTRY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+"),
    re.compile(r"\bdoi(?::|/)", re.IGNORECASE),
)
_PAREN_URL_PATTERN = re.compile(r"\(<(?:https?|mailto):[^>]+>\)", re.IGNORECASE)
_URL_PATTERN = re.compile(
    r"(?:<https?://[^>]+>|https?://\S+|www\.\S+|mailto:\S+)",
    re.IGNORECASE,
)
_URL_QUERY_PATTERN = re.compile(r"\butm_[a-z]+=[^)\s]+", re.IGNORECASE)
_DOMAIN_ARTIFACT_PATTERN = re.compile(
    r"\b(?:nejm|annals|jamanetwork)\s*\.\s*(?:org|com)\b",
    re.IGNORECASE,
)
_REFERENCE_PAREN_PATTERN = re.compile(
    r"\([^)]*\b(?:e?(?:table|figure|figures?)|appendix|supplement(?:ary)?|"
    r"multimedia appendix|available at)\b[^)]*\)",
    re.IGNORECASE,
)
_EMPTY_PARENS_PATTERN = re.compile(r"\(\s*\)")
_BROKEN_SUPPLEMENT_PATTERN = re.compile(r"supple-\s*ment", re.IGNORECASE)
_AVAILABLE_AT_PATTERN = re.compile(
    r"\b(?:available|posted)\s+(?:at|from)\b[^.]*\.?",
    re.IGNORECASE,
)
_TRAILING_REFERENCE_SENTENCE_PATTERN = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:for [^.]{0,120}?\s+)?(?:see\s+)?"
    r"(?:e?(?:table|tables|figure|figures)|appendix|eappendix|supplement(?:ary)?)\b[^.]*\.\s*$",
    re.IGNORECASE,
)
_DETAILS_IN_SUPPLEMENT_PATTERN = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:details?|definitions?|descriptions?)\s+"
    r"(?:are|is)\s+(?:provided|reported|described|in)\b[^.]*\.\s*$",
    re.IGNORECASE,
)
_NEJM_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:disclosure forms|data sharing statement)\b[^.]*\.",
    re.IGNORECASE,
)
# Abbreviation legends ("HR indicates hazard ratio; CI, confidence interval; ...")
# are figure/table glosses that must be excluded from analysis text.  The
# "indicates" branch fires ONLY when an ALL-CAPS acronym (>=2 chars) sits
# immediately before "indicates" (case-sensitive) and "indicates" is not
# followed by "that".  This avoids deleting ordinary prose such as
# "Evidence indicates that ..." / "the literature indicates ...", which the
# earlier IGNORECASE pattern silently dropped (see body_text_loss bug).
_ABBREVIATION_LEGEND_PATTERN = re.compile(
    r"^(?:(?i:abbreviations?\s*:)"
    r"|[A-Za-z0-9/(),;.%+\- ]{0,80}?\b[A-Z][A-Z0-9/()+.\-]+\s+indicates\b(?!\s+that\b))"
)
_SHORT_LABEL_PATTERN = re.compile(r"^[A-Z][A-Za-z.-]*(?:\s+[A-Z][A-Za-z.-]*){0,2}$")
_DROPPED_CAPITAL_RE = re.compile(r"^([A-Z])\s+([a-z])")
_PREAMBLE_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:editorial|related article|invited commentary|comment)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:group information|visual abstract)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:cme quiz|multimedia)\b", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*jama\s*[|·]", re.IGNORECASE),
)
_AFFILIATION_KEYWORD_PATTERN = re.compile(
    r"\b(?:department|division|institute|school of medicine|college of medicine|"
    r"university|hospital|medical center|clinic|faculty|program director|"
    r"assistant professor|associate professor|clinical professor|professor|"
    r"resident(?: physician)?|medical student|student|fellow|chair|director|dean|surgeon)\b",
    re.IGNORECASE,
)
_PERSON_BIO_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*[A-Z](?:\.[A-Z])+\.?\s+[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3}\s+(?:is|was|are)\b"
    ),
    re.compile(r"^\s*[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,4}\s+(?:is|was|are)\b"),)


def _apply_csv_stage2_filters(
    meta: pd.DataFrame,
    title_exclusion_patterns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Apply Stage 2 filters not reflected in the CSV's included column."""
    n_before = len(meta)

    acp_mask = meta["doi"].fillna("").str.lower().str.rstrip(".").str.endswith("-jc")
    n_acp = int(acp_mask.sum())
    if n_acp:
        meta = meta[~acp_mask].copy()
        logger.info("Excluded %d ACP Journal Club synopsis articles", n_acp)

    compiled_title_patterns = compile_journal_patterns(title_exclusion_patterns)
    exclusion_reasons = meta.apply(
        lambda row: metadata_non_original_reason(
            journal_abbreviation=str(row.get("journal_abbreviation") or row.get("journal") or ""),
            title=str(row.get("title") or ""),
            doi=(row.get("doi") or None),
            has_abstract=str(row.get("has_abstract") or "").lower() == "true",
            abstract_text=None,
            compiled_title_patterns=compiled_title_patterns,
        ),
        axis=1,
    )
    non_original_mask = exclusion_reasons.notna()
    n_non_original = int(non_original_mask.sum())
    if n_non_original:
        sample_reasons = exclusion_reasons[non_original_mask].value_counts().to_dict()
        meta = meta.loc[~non_original_mask].copy()
        logger.info("Excluded %d non-original articles from saved CSV", n_non_original)
        logger.info("Saved CSV exclusion reasons: %s", sample_reasons)

    logger.info(
        "CSV Stage 2 filters: %d -> %d (%d excluded)",
        n_before, len(meta), n_before - len(meta),
    )
    return meta


def _load_med_teach_exclusions(csv_path: Path) -> set[str]:
    """Load Med Teach PMIDs to exclude based on article type CSV."""
    if not csv_path.exists():
        logger.warning("Med Teach article types CSV not found: %s", csv_path)
        return set()
    df = pd.read_csv(csv_path, dtype=str)
    excluded = set()
    for _, row in df.iterrows():
        article_type = (row.get("article_type") or "").strip().lower()
        if article_type not in _MED_TEACH_ORIGINAL_TYPES:
            excluded.add(str(row["pmid"]))
    logger.info("Med Teach exclusions loaded: %d PMIDs", len(excluded))
    return excluded


def _apply_final_article_filters(
    meta: pd.DataFrame,
    med_teach_csv: Path | None = None,
    title_exclusion_patterns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Apply all late-stage cohort filters and return the final analysis set."""
    meta = _apply_csv_stage2_filters(meta, title_exclusion_patterns=title_exclusion_patterns)

    jama_mask = meta["pmid"].isin(_JAMA_SPECIAL_COMMUNICATION_PMIDS)
    n_jama_excl = jama_mask.sum()
    if n_jama_excl:
        meta = meta[~jama_mask].copy()
        logger.info("Excluded %d JAMA Special Communication articles", n_jama_excl)

    conf_mask = meta["pmid"].isin(_CONFERENCE_ABSTRACT_PMIDS)
    n_conf_excl = conf_mask.sum()
    if n_conf_excl:
        meta = meta[~conf_mask].copy()
        logger.info("Excluded %d conference-abstract articles", n_conf_excl)

    bad_extract_mask = meta["pmid"].isin(_BAD_EXTRACT_PMIDS)
    n_bad_extract = bad_extract_mask.sum()
    if n_bad_extract:
        meta = meta[~bad_extract_mask].copy()
        logger.info("Excluded %d bad-extract articles", n_bad_extract)

    synopsis_mask = meta["pmid"].isin(_CLINICAL_EVIDENCE_SYNOPSIS_PMIDS)
    n_synopsis = synopsis_mask.sum()
    if n_synopsis:
        meta = meta[~synopsis_mask].copy()
        logger.info("Excluded %d JAMA Clinical Evidence Synopsis articles", n_synopsis)

    if med_teach_csv and med_teach_csv.exists():
        mt_excl = _load_med_teach_exclusions(med_teach_csv)
        mt_mask = meta["pmid"].isin(mt_excl)
        n_mt_excl = mt_mask.sum()
        if n_mt_excl:
            meta = meta[~mt_mask].copy()
            logger.info("Excluded %d Med Teach non-original articles", n_mt_excl)

    logger.info("Final article count: %d", len(meta))
    return meta


def load_final_metadata(
    articles_csv: Path,
    med_teach_csv: Path | None = None,
) -> pd.DataFrame:
    """Load the definitive analysis manifest from the raw included CSV."""
    meta = pd.read_csv(articles_csv, dtype=str)
    logger.info("Loaded metadata: %d rows", len(meta))

    if "included" in meta.columns:
        included_mask = meta["included"].fillna("").str.lower() == "true"
        meta = meta[included_mask].copy()
        logger.info("After included=True filter: %d rows", len(meta))

    pubmed_config = get_pubmed_config()
    meta = _apply_final_article_filters(
        meta,
        med_teach_csv=med_teach_csv,
        title_exclusion_patterns=pubmed_config.get("title_exclusion_patterns", {}),
    )
    return meta.reset_index(drop=True)


def _load_cleaned_texts(cleaned_dir: Path) -> dict[str, PmcFullText]:
    """Load all cleaned-text JSON files into a dict keyed by PMID."""
    texts: dict[str, PmcFullText] = {}
    if not cleaned_dir.exists():
        logger.warning("Cleaned text directory not found: %s", cleaned_dir)
        return texts
    json_files = list(cleaned_dir.glob("*.json"))
    logger.info("Loading %d cleaned-text JSON files...", len(json_files))
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            ft = PmcFullText(**data)
            texts[ft.pmid] = ft
        except Exception:
            logger.warning("Failed to load %s", jf.name, exc_info=True)
    logger.info("Loaded %d cleaned-text records", len(texts))
    return texts


# ---------------------------------------------------------------------------
# Section reclassification
# ---------------------------------------------------------------------------

def _reclassify_sections(
    ft: PmcFullText,
    journal_patterns: list[tuple[re.Pattern[str], str]] | None = None,
) -> bool:
    """Re-apply title patterns to fix misclassified sections.

    Handles two cases:
    1. ``"other"`` sections whose title now matches an IMRaD pattern
       (e.g. Perspect Med Educ "Implementation", "Evaluation", "Critical
       Reflection").
    2. ``"results"`` sections whose title matches a *discussion* pattern that
       is more specific than the generic ``findings`` pattern (e.g. JMIR
       "Principal Findings", "Interpretation of Findings").

    *journal_patterns* (if provided) are checked **before** the global
    ``_TITLE_PATTERNS``, giving journal-specific overrides higher priority.

    Mutates *ft.sections* in-place.  Returns ``True`` if any section was
    reclassified.
    """
    changed = False
    for sec in ft.sections:
        if sec.section_type == "abstract" or not sec.section_title:
            continue
        # 1. Journal-specific patterns (highest priority)
        new_type = "other"
        if journal_patterns:
            for pattern, label in journal_patterns:
                if pattern.search(sec.section_title):
                    new_type = label
                    break
        # 2. Global patterns (fallback)
        if new_type == "other":
            for pattern, label in _TITLE_PATTERNS:
                if pattern.search(sec.section_title):
                    new_type = label
                    break
        if new_type == sec.section_type or new_type == "other":
            continue
        # Accept reclassification for "other" → any IMRaD type
        if sec.section_type == "other":
            sec.section_type = new_type
            changed = True
        # Accept results → discussion (JMIR "Principal Findings" etc.)
        elif sec.section_type == "results" and new_type == "discussion":
            sec.section_type = new_type
            changed = True
    return changed


def _detect_non_imrad(ft: PmcFullText) -> tuple[bool, str | None]:
    """Flag articles that should be handled as non-IMRaD.

    Heuristics (checked in order):
    1. Missing both methods AND results → ``"no_methods_and_results"``
    2. Missing both results AND discussion → ``"no_results_and_discussion"``
       (catches short communications, innovation reports, etc.)
    """
    body_sections = [sec for sec in ft.sections if sec.section_type != "abstract"]
    if not body_sections:
        return False, None

    has_methods = any(sec.section_type == "methods" for sec in body_sections)
    has_results = any(sec.section_type == "results" for sec in body_sections)
    has_discussion = any(sec.section_type == "discussion" for sec in body_sections)
    if not has_methods and not has_results:
        return True, "no_methods_and_results"
    if not has_results and not has_discussion:
        return True, "no_results_and_discussion"
    return False, None


# ---------------------------------------------------------------------------
# Manual per-article overrides
# ---------------------------------------------------------------------------


class SectionOverrideAction(BaseModel):
    """A single action to apply to an article's sections."""

    action: str = Field(
        description="'rename_section' | 'set_non_imrad' | 'split_section'"
    )
    from_type: str | None = Field(default=None)
    title_contains: str | None = Field(default=None)
    # Exact, CASE-SENSITIVE title match. Use to disambiguate sections whose
    # titles differ only by case (e.g. a long 'CONCLUSIONS' discussion vs a
    # short 'Conclusions' wrap-up), which title_contains (case-insensitive)
    # cannot tell apart.
    title_equals: str | None = Field(default=None)
    to_type: str | None = Field(default=None)
    reason: str | None = Field(default=None)
    # split_section parameters: split the matched section's text at *split_at*
    # (a verbatim anchor substring) and re-type the tail (from the anchor
    # onward) as *tail_to_type*. The head (before the anchor) keeps the
    # original type. Used to recover a Discussion that the parser merged into
    # the tail of a Results/Conclusion section with no detectable heading.
    split_at: str | None = Field(default=None)
    tail_to_type: str | None = Field(default="discussion")
    tail_title: str | None = Field(default=None)


class ArticleOverride(BaseModel):
    """Manual override specification for one PMID."""

    pmid: str
    journal: str = ""
    reason: str = ""
    actions: list[SectionOverrideAction]


def load_section_overrides(toml_path: Path) -> dict[str, list[SectionOverrideAction]]:
    """Load per-article overrides from a TOML file.

    Returns ``{pmid: [SectionOverrideAction, ...]}`` mapping.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    overrides_raw = data.get("overrides", [])
    result: dict[str, list[SectionOverrideAction]] = {}
    for entry in overrides_raw:
        ao = ArticleOverride(**entry)
        actions = ao.actions
        if actions:
            result[ao.pmid] = actions
    return result


def _apply_manual_overrides(
    ft: PmcFullText,
    actions: list[SectionOverrideAction],
) -> tuple[bool, str | None]:
    """Apply manual section overrides to a PmcFullText object.

    Returns ``(changed, non_imrad_reason)`` where *non_imrad_reason* is set
    only when a ``set_non_imrad`` action is present.
    """
    changed = False
    non_imrad_reason: str | None = None

    for act in actions:
        if act.action == "rename_section":
            for sec in ft.sections:
                if act.from_type and sec.section_type != act.from_type:
                    continue
                if act.title_equals is not None and sec.section_title != act.title_equals:
                    continue
                if act.title_contains and act.title_contains.lower() not in sec.section_title.lower():
                    continue
                if act.to_type and sec.section_type != act.to_type:
                    sec.section_type = act.to_type
                    changed = True
        elif act.action == "split_section":
            _apply_split_section(ft, act)
            changed = True
        elif act.action == "set_non_imrad":
            non_imrad_reason = act.reason or "manual_override"
            changed = True

    return changed, non_imrad_reason


def _apply_split_section(ft: PmcFullText, act: SectionOverrideAction) -> None:
    """Split one section's text at *act.split_at*, re-typing the tail.

    The matched section (``from_type`` + ``title_contains``) is divided at the
    first occurrence of the verbatim anchor ``split_at``: the head keeps its
    original type, while the tail (anchor onward) becomes a new section of type
    ``tail_to_type`` inserted immediately after. Whitespace is normalised only
    for matching — the anchor must appear verbatim (after collapsing runs of
    whitespace) exactly once across the candidate sections, otherwise a
    ``ValueError`` is raised so that corpus builds fail loudly rather than
    silently mis-splitting.
    """
    if not act.split_at:
        raise ValueError(
            f"split_section for PMID {ft.pmid} requires a non-empty 'split_at' anchor"
        )

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    anchor = _norm(act.split_at)
    matches: list[tuple[int, int]] = []  # (section_index, char_offset_in_norm_text)
    for idx, sec in enumerate(ft.sections):
        if act.from_type and sec.section_type != act.from_type:
            continue
        if act.title_equals is not None and sec.section_title != act.title_equals:
            continue
        if act.title_contains and act.title_contains.lower() not in sec.section_title.lower():
            continue
        norm_text = _norm(sec.text)
        pos = norm_text.find(anchor)
        if pos != -1:
            # Reject ambiguity: anchor must be unique within this section too.
            if norm_text.find(anchor, pos + 1) != -1:
                raise ValueError(
                    f"split_section anchor for PMID {ft.pmid} occurs more than once "
                    f"within section [{idx}] {sec.section_type!r} '{sec.section_title}'. "
                    f"Use a longer, unique anchor."
                )
            matches.append((idx, pos))

    if len(matches) != 1:
        raise ValueError(
            f"split_section anchor for PMID {ft.pmid} matched {len(matches)} sections "
            f"(expected exactly 1). Anchor={anchor[:60]!r}, from_type={act.from_type!r}, "
            f"title_contains={act.title_contains!r}."
        )

    sec_idx, _ = matches[0]
    sec = ft.sections[sec_idx]
    # Locate the anchor in the ORIGINAL (un-normalised) text so the split keeps
    # the real characters. Build a whitespace-tolerant regex from the anchor.
    anchor_re = re.compile(
        r"\s+".join(re.escape(tok) for tok in anchor.split(" ")),
        re.DOTALL,
    )
    m = anchor_re.search(sec.text)
    if m is None:  # pragma: no cover - normalised match guarantees this exists
        raise ValueError(
            f"split_section anchor for PMID {ft.pmid} normalised-matched but could "
            f"not be located in the raw section text."
        )

    head_text = sec.text[: m.start()].rstrip()
    tail_text = sec.text[m.start():].strip()
    if not head_text or not tail_text:
        raise ValueError(
            f"split_section for PMID {ft.pmid} produced an empty head or tail; "
            f"anchor is at a section boundary."
        )

    sec.text = head_text
    sec.word_count = len(head_text.split())
    tail_sec = PmcSection(
        section_type=act.tail_to_type or "discussion",
        section_title=act.tail_title or "",
        text=tail_text,
        word_count=len(tail_text.split()),
    )
    ft.sections.insert(sec_idx + 1, tail_sec)


def _looks_like_author_segment(text: str) -> bool:
    """Heuristic detection of an author-name segment."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned or _CONTENT_CUE_PATTERN.search(cleaned):
        return False
    if not _AUTHOR_DEGREE_PATTERN.search(cleaned):
        return False

    name_part = cleaned.split(",", 1)[0].strip()
    tokens = [tok for tok in re.split(r"\s+", name_part) if tok]
    if not 2 <= len(tokens) <= 6:
        return False

    particles = {"al", "bin", "da", "de", "del", "der", "di", "la", "le", "van", "von"}
    for token in tokens:
        bare = token.strip(".,")
        if bare.lower() in particles:
            continue
        if not re.match(r"^[A-Z][\w'’.-]*$", bare):
            return False
    return True


def _looks_like_author_line(text: str) -> bool:
    """Detect semicolon-delimited author bylines that leaked into text."""
    segments = [seg.strip() for seg in text.split(";") if seg.strip()]
    if len(segments) < 2:
        return False
    return sum(_looks_like_author_segment(seg) for seg in segments) >= 2


def _looks_like_affiliation_paragraph(text: str) -> bool:
    """Detect author-affiliation paragraphs copied from article front/back matter."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return False

    # Affiliation paragraphs are short; long paragraphs are body text.
    # Prevents false positives like "Tolebrutinib is..." matching as "Name is...".
    if len(cleaned.split()) > 60:
        return False

    lower = cleaned.lower()
    if "orcid" in lower:
        return True

    if any(pattern.match(cleaned) for pattern in _PERSON_BIO_PREFIX_PATTERNS):
        return bool(
            _AFFILIATION_KEYWORD_PATTERN.search(cleaned)
            or "@" in cleaned
            or "e-mail" in lower
        )

    return False


def _looks_like_reference_entry(text: str) -> bool:
    """Detect stray bibliographic entries that should not enter analysis."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned.split()) > 80:
        return False
    if _REFERENCE_ENTRY_PATTERNS[0].match(cleaned):
        return True
    if (
        _REFERENCE_ENTRY_PATTERNS[1].search(cleaned)
        and re.search(r"\b(?:19|20)\d{2}\b", cleaned)
    ):
        return True
    return False


def _strip_inline_analysis_artifacts(text: str) -> str | None:
    """Remove inline URL/supplement boilerplate while preserving prose."""
    cleaned = text
    cleaned = _PAREN_URL_PATTERN.sub(" ", cleaned)
    cleaned = _URL_QUERY_PATTERN.sub(" ", cleaned)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _DOMAIN_ARTIFACT_PATTERN.sub(" ", cleaned)
    cleaned = _BROKEN_SUPPLEMENT_PATTERN.sub("supplement", cleaned)

    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _REFERENCE_PAREN_PATTERN.sub(" ", cleaned)
        cleaned = _EMPTY_PARENS_PATTERN.sub(" ", cleaned)

    cleaned = _TRAILING_REFERENCE_SENTENCE_PATTERN.sub(" ", cleaned)
    cleaned = _DETAILS_IN_SUPPLEMENT_PATTERN.sub(" ", cleaned)
    cleaned = _AVAILABLE_AT_PATTERN.sub(" ", cleaned)
    cleaned = _NEJM_DISCLOSURE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\bORCID\s*:\s*.*$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([([])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" \t\n,;:")
    return cleaned or None


def _is_analysis_noise_paragraph(text: str) -> bool:
    """Return True for non-body paragraphs to exclude from analysis text."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return True
    if _looks_like_author_line(cleaned):
        return True
    if _looks_like_affiliation_paragraph(cleaned):
        return True
    if _CAPTION_LABEL_PATTERN.match(cleaned) and not _looks_like_figure_reference_prose(
        cleaned
    ):
        return True
    if _ABBREVIATION_LEGEND_PATTERN.match(cleaned):
        return True
    if len(cleaned.split()) <= 3 and _SHORT_LABEL_PATTERN.match(cleaned):
        return True
    if _looks_like_reference_entry(cleaned):
        return True
    return any(pattern.search(cleaned) for pattern in _ANALYSIS_METADATA_PATTERNS)


def _fix_dropped_capital(text: str) -> str:
    """Fix PDF dropped-capital artifact: 'N ewborn' -> 'Newborn'."""
    return _DROPPED_CAPITAL_RE.sub(lambda m: m.group(1) + m.group(2), text)


def _is_preamble_noise(text: str) -> bool:
    """Return True for paragraphs that are noise in a preamble/abstract section.

    Catches short link/label paragraphs and JAMA-specific metadata lines
    that ``_is_analysis_noise_paragraph`` may not cover.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return True
    words = cleaned.split()
    # Very short paragraphs dominated by URLs are noise
    if len(words) <= 12 and _URL_PATTERN.search(cleaned):
        return True
    if any(pat.match(cleaned) for pat in _PREAMBLE_NOISE_PATTERNS):
        return True
    return False


def extract_introduction_from_preamble(
    sections: list[PmcSection],
    pubmed_abstract: str,
    min_intro_words: int = 30,
) -> list[PmcSection]:
    """Extract introduction text trapped in the abstract section.

    JAMA PDFs often place author names and introduction prose in the
    preamble (parsed as "abstract") while the real structured abstract
    is only available from PubMed.  This function:

    1. Splits the parsed abstract into paragraphs.
    2. Filters out noise (author lines, affiliations, URLs, metadata).
    3. Reclassifies remaining prose as "introduction".
    4. Replaces the parsed abstract with the authoritative PubMed text.

    Returns the *same* list object when no change is made (caller can
    use ``is`` to detect modifications).
    """
    if not pubmed_abstract or not pubmed_abstract.strip():
        return sections

    # Skip if introduction already exists
    if any(s.section_type == "introduction" for s in sections):
        return sections

    # Find abstract section(s)
    abstract_indices = [
        i for i, s in enumerate(sections) if s.section_type == "abstract"
    ]
    if not abstract_indices:
        return sections

    # Collect all abstract text
    abstract_texts = [sections[i].text for i in abstract_indices]
    combined_abstract = "\n\n".join(abstract_texts)

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", combined_abstract) if p.strip()]

    # Filter noise and collect prose
    prose_parts: list[str] = []
    for para in paragraphs:
        if _is_analysis_noise_paragraph(para):
            continue
        if _is_preamble_noise(para):
            continue
        # Strip inline artifacts (URLs, supplements, etc.)
        cleaned = _strip_inline_analysis_artifacts(para)
        if not cleaned:
            continue
        if len(cleaned.split()) < 10:
            continue
        # Fix dropped-capital artifact
        cleaned = _fix_dropped_capital(cleaned)
        prose_parts.append(cleaned)

    # Check if we have enough prose to form an introduction
    total_words = sum(len(p.split()) for p in prose_parts)
    if total_words < min_intro_words:
        return sections

    # Build new sections list
    intro_text = "\n\n".join(prose_parts)
    new_abstract = PmcSection(
        section_type="abstract",
        section_title="Abstract",
        text=pubmed_abstract.strip(),
        word_count=len(pubmed_abstract.split()),
    )
    new_intro = PmcSection(
        section_type="introduction",
        section_title="Introduction",
        text=intro_text,
        word_count=total_words,
    )

    # Replace abstract sections and insert introduction
    new_sections: list[PmcSection] = [new_abstract, new_intro]
    for i, sec in enumerate(sections):
        if i not in abstract_indices:
            new_sections.append(sec)

    return new_sections


def _sanitize_section_text(text: str) -> str | None:
    """Remove analysis-irrelevant paragraphs from one section."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    kept = []
    for part in paragraphs:
        cleaned = _strip_inline_analysis_artifacts(part)
        if not cleaned or _is_analysis_noise_paragraph(cleaned):
            continue
        kept.append(cleaned)
    if not kept:
        return None
    return "\n\n".join(kept)


def _sanitize_sections_for_analysis(ft: PmcFullText) -> PmcFullText:
    """Drop author lines, captions, and bibliographic metadata from sections."""
    cleaned_sections = []
    for sec in ft.sections:
        clean_text = _sanitize_section_text(sec.text)
        if not clean_text:
            continue
        cleaned_sections.append(
            PmcSection(
                section_type=sec.section_type,
                section_title=sec.section_title,
                text=clean_text,
                word_count=len(clean_text.split()),
            )
        )

    return ft.model_copy(
        update={
            "sections": cleaned_sections,
            "total_word_count": sum(sec.word_count for sec in cleaned_sections),
            "has_body": any(sec.section_type != "abstract" for sec in cleaned_sections),
        }
    )


def _sections_to_columns(ft: PmcFullText) -> dict[str, Any]:
    """Flatten PmcFullText sections into IMRaD column values."""
    buckets: dict[str, list[str]] = {col: [] for col in IMRAD_COLUMNS}
    buckets["other"] = []

    for sec in ft.sections:
        st = sec.section_type.lower()
        if st in buckets:
            buckets[st].append(sec.text)
        else:
            buckets["other"].append(sec.text)

    result: dict[str, Any] = {}
    for col in IMRAD_COLUMNS:
        result[col] = "\n\n".join(buckets[col]) if buckets[col] else None
    result["other_sections"] = (
        "\n\n".join(buckets["other"]) if buckets["other"] else None
    )

    # body_full = everything except abstract
    body_parts = []
    for col in ("introduction", "methods", "results", "discussion", "conclusion"):
        if result[col]:
            body_parts.append(result[col])
    if result["other_sections"]:
        body_parts.append(result["other_sections"])
    result["body_full"] = "\n\n".join(body_parts) if body_parts else None
    result["citation_tables_figures"] = ft.citation_tables_figures or None
    result["has_body"] = bool(body_parts)
    result["word_count"] = sum(sec.word_count for sec in ft.sections)
    result["source_format"] = ft.source_type
    is_non_imrad, reason = _detect_non_imrad(ft)
    result["is_non_imrad"] = is_non_imrad
    result["non_imrad_reason"] = reason
    return result


def _apply_pubmed_abstract_filters(
    df: pd.DataFrame,
    abstract_exclusion_patterns: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Drop non-original articles identified from PubMed abstract phrasing."""
    compiled_abstract_patterns = compile_journal_patterns(abstract_exclusion_patterns)
    if not compiled_abstract_patterns or "abstract_pubmed" not in df.columns:
        return df

    def _opt_str(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return str(value) or None

    exclusion_reasons = df.apply(
        lambda row: metadata_non_original_reason(
            journal_abbreviation=_opt_str(row.get("journal")) or "",
            title=_opt_str(row.get("title")) or "",
            doi=_opt_str(row.get("doi")),
            has_abstract=bool(_opt_str(row.get("abstract_pubmed"))),
            abstract_text=_opt_str(row.get("abstract_pubmed")),
            compiled_abstract_patterns=compiled_abstract_patterns,
        ),
        axis=1,
    )
    mask = exclusion_reasons.notna() & exclusion_reasons.str.startswith("Excluded abstract pattern:")
    n_excluded = int(mask.sum())
    if n_excluded:
        logger.info(
            "Excluded %d articles by PubMed abstract rules: %s",
            n_excluded,
            exclusion_reasons[mask].value_counts().to_dict(),
        )
        return df.loc[~mask].reset_index(drop=True)
    return df


def build_corpus(
    articles_csv: Path,
    cleaned_text_dir: Path,
    med_teach_csv: Path | None = None,
    pubmed_abstracts: dict[str, str] | None = None,
    overrides_toml: Path | None = None,
) -> pd.DataFrame:
    """Build the analysis corpus DataFrame.

    Args:
        articles_csv: Path to articles_2025_included.csv.
        cleaned_text_dir: Path to data/interim/cleaned_text/.
        med_teach_csv: Path to med_teach_article_types.csv (optional).
        pubmed_abstracts: {pmid: abstract_text} from PubMed API (optional).
        overrides_toml: Path to section_overrides.toml (optional).

    Returns:
        DataFrame ready for Parquet export.
    """
    # 1. Load metadata
    meta = load_final_metadata(articles_csv, med_teach_csv=med_teach_csv)

    # 2. Load cleaned texts
    texts = _load_cleaned_texts(cleaned_text_dir)

    # Step 0: Reclassify sections with updated title patterns
    # Build per-journal pattern map from TOML config
    remediation = get_remediation_config()
    journal_title_patterns_raw = remediation.get("journal_title_patterns", {})
    journal_compiled: dict[str, list[tuple[re.Pattern[str], str]]] = {}
    for jname, patterns in journal_title_patterns_raw.items():
        compiled = []
        for p in patterns:
            compiled.append((re.compile(p["pattern"]), p["section_type"]))
        journal_compiled[jname] = compiled

    # Build {pmid: journal} lookup from metadata
    pmid_journal: dict[str, str] = {
        str(row["pmid"]): str(row.get("journal") or "")
        for _, row in meta.iterrows()
    }

    n_reclassified = 0
    for pmid, ft in texts.items():
        j_pats = journal_compiled.get(pmid_journal.get(pmid, ""))
        if _reclassify_sections(ft, journal_patterns=j_pats):
            n_reclassified += 1
    if n_reclassified:
        logger.info("Reclassified sections for %d articles", n_reclassified)

    # Step 1: Anchor abstract/body boundary using PubMed abstract (most precise)
    if pubmed_abstracts:
        n_anchored = 0
        for pmid, ft in texts.items():
            pa = pubmed_abstracts.get(pmid)
            if pa:
                new_sections = anchor_body_start_with_pubmed(ft.sections, pa)
                if new_sections is not ft.sections:
                    ft.sections = new_sections
                    n_anchored += 1
        if n_anchored:
            logger.info(
                "Anchored abstract/body boundary for %d articles", n_anchored
            )

    # Step 1.5: Extract introduction from preamble content in abstract sections
    # (handles JAMA PDFs where author names + intro prose are parsed as "abstract")
    if pubmed_abstracts:
        n_preamble_intro = 0
        for pmid, ft in texts.items():
            pa = pubmed_abstracts.get(pmid)
            if pa:
                new_sections = extract_introduction_from_preamble(ft.sections, pa)
                if new_sections is not ft.sections:
                    ft.sections = new_sections
                    n_preamble_intro += 1
        if n_preamble_intro:
            logger.info(
                "Extracted introduction from preamble for %d articles",
                n_preamble_intro,
            )

    # Step 1.75: Force-split oversized abstract sections that swallowed body
    # text even though an introduction exists (anchor step skips those).
    if pubmed_abstracts:
        n_forced = 0
        for pmid, ft in texts.items():
            pa = pubmed_abstracts.get(pmid)
            if not pa:
                continue
            abs_wc = sum(
                s.word_count for s in ft.sections if s.section_type == "abstract"
            )
            pa_wc = len(pa.split())
            if abs_wc <= max(600, pa_wc * 1.5 + 150):
                continue
            new_sections = anchor_body_start_with_pubmed(
                ft.sections, pa, force_with_existing_intro=True
            )
            if new_sections is not ft.sections:
                ft.sections = new_sections
                n_forced += 1
        if n_forced:
            logger.info(
                "Force-split oversized abstract for %d articles", n_forced
            )

    # Step 2: Fallback — infer introduction from position for remaining articles
    n_inferred = 0
    for pmid, ft in texts.items():
        new_sections = infer_introduction_from_position(ft.sections)
        if new_sections is not ft.sections:
            ft.sections = new_sections
            n_inferred += 1
    if n_inferred:
        logger.info("Inferred introduction from position for %d articles", n_inferred)

    # Step 2.2: Recover results from position for numbered-IMRaD articles whose
    # "RESULTS" heading was mangled by PDF letter-spacing ("RE SULTS").
    n_results_inferred = 0
    for pmid, ft in texts.items():
        new_sections = infer_results_from_position(ft.sections)
        if new_sections is not ft.sections:
            ft.sections = new_sections
            n_results_inferred += 1
    if n_results_inferred:
        logger.info(
            "Recovered results from position for %d articles", n_results_inferred
        )

    # Step 2.5: Apply manual per-article overrides (before sanitisation)
    manual_overrides: dict[str, list[SectionOverrideAction]] = {}
    override_non_imrad: dict[str, str] = {}  # pmid → reason (from set_non_imrad)
    if overrides_toml and overrides_toml.exists():
        manual_overrides = load_section_overrides(overrides_toml)
        logger.info("Loaded manual overrides for %d articles", len(manual_overrides))
    if manual_overrides:
        n_overridden = 0
        for pmid, ft in texts.items():
            if pmid in manual_overrides:
                changed, ni_reason = _apply_manual_overrides(ft, manual_overrides[pmid])
                if changed:
                    n_overridden += 1
                if ni_reason:
                    override_non_imrad[pmid] = ni_reason
        if n_overridden:
            logger.info("Applied manual overrides for %d articles", n_overridden)

    # Remove author lines, bibliographic metadata, and figure/table labels
    # from the analysis text while keeping the raw cleaned_text JSON intact.
    texts = {
        pmid: _sanitize_sections_for_analysis(ft)
        for pmid, ft in texts.items()
    }

    # 3. Build rows
    rows: list[dict[str, Any]] = []
    for _, m in meta.iterrows():
        pmid = str(m["pmid"])
        authors_str = m.get("authors", "")
        if pd.isna(authors_str):
            authors_str = ""
        n_authors = len([a for a in authors_str.split(";") if a.strip()]) if authors_str else 0

        row: dict[str, Any] = {
            "pmid": pmid,
            "doi": m.get("doi") if pd.notna(m.get("doi")) else None,
            "title": m.get("title", "") if pd.notna(m.get("title")) else "",
            "journal": m.get("journal", "") if pd.notna(m.get("journal")) else "",
            "journal_field": m.get("journal_field", "") if pd.notna(m.get("journal_field")) else "",
            "pub_year": int(m["pub_year"]) if pd.notna(m.get("pub_year")) else None,
            "pub_date": m.get("pub_date") if pd.notna(m.get("pub_date")) else None,
            "authors": authors_str,
            "n_authors": n_authors,
            "publication_types": (
                m.get("publication_types", "")
                if pd.notna(m.get("publication_types"))
                else ""
            ),
            "pmcid": m.get("pmcid") if pd.notna(m.get("pmcid")) else None,
        }

        # PubMed abstract (authoritative source)
        row["abstract_pubmed"] = (
            pubmed_abstracts.get(pmid) if pubmed_abstracts else None
        )

        # Merge text columns
        if pmid in texts:
            row.update(_sections_to_columns(texts[pmid]))
        else:
            for col in IMRAD_COLUMNS:
                row[col] = None
            row["other_sections"] = None
            row["body_full"] = None
            row["citation_tables_figures"] = None
            row["is_non_imrad"] = False
            row["non_imrad_reason"] = None
            row["has_body"] = False
            row["word_count"] = 0
            row["source_format"] = None

        # Apply manual non-IMRaD overrides
        if manual_overrides and pmid in override_non_imrad:
            row["is_non_imrad"] = True
            row["non_imrad_reason"] = override_non_imrad[pmid]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Exclude short reports from the final analysis corpus.
    short_mask = df["has_body"] & (df["word_count"] < SHORT_ARTICLE_THRESHOLD)
    n_short = int(short_mask.sum())
    if n_short:
        short_pmids = df.loc[short_mask, "pmid"].astype(str).tolist()
        preview = ", ".join(short_pmids[:10])
        suffix = " ..." if len(short_pmids) > 10 else ""
        logger.info(
            "Excluded %d short articles (< %d words): %s%s",
            n_short,
            SHORT_ARTICLE_THRESHOLD,
            preview,
            suffix,
        )
        df = df.loc[~short_mask].reset_index(drop=True)

    pubmed_config = get_pubmed_config()
    df = _apply_pubmed_abstract_filters(
        df,
        pubmed_config.get("abstract_exclusion_patterns", {}),
    )

    # Keep string columns as object dtype (polars handles categorization itself)
    df["has_body"] = df["has_body"].astype(bool)
    df["is_non_imrad"] = df["is_non_imrad"].astype(bool)

    # Ensure integer types (use numpy-backed types for polars compatibility)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").fillna(0).astype("int64")
    df["n_authors"] = df["n_authors"].astype("int64")
    df["word_count"] = df["word_count"].astype("int64")

    return df


def apply_screening_filter(
    df: pd.DataFrame,
    screening_csv: Path,
    review_tsv_paths: list[Path] | None = None,
) -> pd.DataFrame:
    """Apply AI screening results + human review overrides.

    For each PMID the final decision is determined as:
    1. If a human review TSV contains ``review_status == "done"`` for that PMID,
       use the human-corrected decision (agree → keep AI decision,
       disagree → use ``review_corrected_decision``).
    2. Otherwise fall back to the AI ``decision`` column in *screening_csv*.

    Only articles with a final decision of ``"include"`` are kept.
    """
    screening = pd.read_csv(screening_csv, dtype=str)
    ai_decisions: dict[str, str] = dict(
        zip(screening["pmid"].astype(str), screening["decision"])
    )

    # Collect human overrides from review TSVs
    human_decisions: dict[str, str] = {}
    for tsv_path in review_tsv_paths or []:
        if not tsv_path.exists():
            continue
        review = pd.read_csv(tsv_path, sep="\t", dtype=str)
        for _, row in review.iterrows():
            if row.get("review_status") != "done":
                continue
            pmid = str(row["pmid"])
            rd = row.get("review_decision", "")
            if rd == "agree":
                # Human agrees with AI → keep AI decision
                human_decisions[pmid] = ai_decisions.get(pmid, "exclude")
            elif rd == "disagree":
                # Human overrides → use corrected decision
                human_decisions[pmid] = row.get("review_corrected_decision", "include")

    # Build final decision map: human override > AI
    def _final_decision(pmid: str) -> str:
        if pmid in human_decisions:
            return human_decisions[pmid]
        return ai_decisions.get(pmid, "include")  # not screened → keep

    mask = df["pmid"].astype(str).map(_final_decision) == "include"
    n_before = len(df)
    df_filtered = df.loc[mask].reset_index(drop=True)
    n_removed = n_before - len(df_filtered)
    logger.info(
        "Screening filter: %d → %d (%d excluded)",
        n_before,
        len(df_filtered),
        n_removed,
    )
    return df_filtered


def save_corpus(df: pd.DataFrame, output_path: Path) -> None:
    """Write corpus DataFrame to Parquet with ZSTD compression.

    Uses polars for Parquet I/O since pyarrow has no ARM64 Windows wheel.
    Builds polars DataFrame from raw dicts to avoid pandas Arrow dtype issues.
    """
    import polars as pl

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build polars DataFrame from records to bypass pandas Arrow dtype issues
    records = df.to_dict(orient="records")
    # Convert NaN/NaT to None for polars
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None

    pl_df = pl.DataFrame(records, infer_schema_length=len(records) or None)
    pl_df.write_parquet(output_path, compression="zstd")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved corpus: %s (%.1f MB, %d rows)", output_path, size_mb, len(df))


def save_article_manifest(meta: pd.DataFrame, output_path: Path) -> None:
    """Write the definitive included-articles manifest CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export = meta.copy()
    if "included" in export.columns:
        export["included"] = "True"
    if "exclusion_reason" in export.columns:
        export["exclusion_reason"] = ""

    export.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved article manifest: %s (%d rows)", output_path, len(export))
