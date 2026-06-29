"""Regex-based detector for deontic modal expressions.

Vocabulary lists come from ``config/default_settings.toml [detection]``.
Regex compilation, group mapping, and priority ordering are handled here.

Detection algorithm
-------------------
1. Normalise the input text (label stripping, quote & whitespace normalisation).
2. Compile all patterns: negation forms first (longest-first), then affirmative.
3. Collect all candidate spans via ``re.finditer``.
4. Sort by start position; greedily select non-overlapping matches.
5. For each accepted match extract *word-level* context (N words before / after).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl

from normative_discourse.config.settings import get_detection_config
from normative_discourse.preprocessing.text_normalizer import (
    count_words,
    normalize_for_detection,
)
from normative_discourse.schemas import (
    EXPRESSION_GROUPS,
    MatchRecord,
    SectionInventoryRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal pattern definitions
# ---------------------------------------------------------------------------
# Each entry maps a compiled regex → (expression_group, canonical_form, is_negation).
# Within a priority tier the *longest* pattern text is checked first so that
# "does not need to" beats "need to".

_APOSTROPHE = r"['\u2019\u02BC]"  # ASCII + RIGHT SINGLE + MODIFIER LETTER

# Negation patterns (checked BEFORE affirmative to avoid double-counting).
_NEGATION_PATTERNS: list[tuple[str, str, str]] = [
    # --- must ---
    (r"\bmust\s+not\b", "must", "must not"),
    (rf"\bmustn{_APOSTROPHE}t\b", "must", "must not"),
    # --- should ---
    (r"\bshould\s+not\b", "should", "should not"),
    (rf"\bshouldn{_APOSTROPHE}t\b", "should", "should not"),
    # --- need_to (longer first) ---
    (rf"\bdoes\s+not\s+need\s+to\b", "need_to", "does not need to"),
    (rf"\bdoesn{_APOSTROPHE}t\s+need\s+to\b", "need_to", "does not need to"),
    (r"\bdo\s+not\s+need\s+to\b", "need_to", "do not need to"),
    (rf"\bdon{_APOSTROPHE}t\s+need\s+to\b", "need_to", "do not need to"),
    (r"\bdid\s+not\s+need\s+to\b", "need_to", "did not need to"),
    (rf"\bdidn{_APOSTROPHE}t\s+need\s+to\b", "need_to", "did not need to"),
    (r"\bneed\s+not\b", "need_to", "need not"),
    (rf"\bneedn{_APOSTROPHE}t\b", "need_to", "need not"),
    # --- have_to (longer first) ---
    (r"\bdoes\s+not\s+have\s+to\b", "have_to", "does not have to"),
    (rf"\bdoesn{_APOSTROPHE}t\s+have\s+to\b", "have_to", "does not have to"),
    (r"\bdo\s+not\s+have\s+to\b", "have_to", "do not have to"),
    (rf"\bdon{_APOSTROPHE}t\s+have\s+to\b", "have_to", "do not have to"),
    (r"\bdid\s+not\s+have\s+to\b", "have_to", "did not have to"),
    (rf"\bdidn{_APOSTROPHE}t\s+have\s+to\b", "have_to", "did not have to"),
    # --- required_to ---
    (r"\bnot\s+required\s+to\b", "required_to", "not required to"),
]

# Affirmative patterns (checked AFTER negation patterns).
_AFFIRMATIVE_PATTERNS: list[tuple[str, str, str]] = [
    # --- must ---  (exclude "mustn" to avoid partial match inside contraction)
    (r"\bmust\b(?!n)", "must", "must"),
    # --- should ---
    (r"\bshould\b(?!n)", "should", "should"),
    # --- need_to ---
    (r"\bneeds?\s+to\b", "need_to", "need to"),
    # --- have_to --- (including "got to" variants)
    (r"\bhave\s+got\s+to\b", "have_to", "have got to"),
    (r"\bhas\s+got\s+to\b", "have_to", "has got to"),
    (r"\bhad\s+got\s+to\b", "have_to", "had got to"),
    (r"\bhave\s+to\b", "have_to", "have to"),
    (r"\bhas\s+to\b", "have_to", "has to"),
    (r"\bhad\s+to\b", "have_to", "had to"),
    # --- required_to ---
    (r"\brequired\s+to\b", "required_to", "required to"),
]


class _CompiledPattern:
    """A compiled regex with its metadata."""

    __slots__ = ("regex", "group", "canonical", "is_negation")

    def __init__(
        self,
        regex: re.Pattern[str],
        group: str,
        canonical: str,
        is_negation: bool,
    ) -> None:
        self.regex = regex
        self.group = group
        self.canonical = canonical
        self.is_negation = is_negation


# ---------------------------------------------------------------------------
# DeonticDetector
# ---------------------------------------------------------------------------

class DeonticDetector:
    """Regex-based deontic expression detector.

    Parameters
    ----------
    context_window : int
        Number of *words* to capture before and after each match.
    """

    def __init__(self, context_window: int | None = None) -> None:
        cfg = get_detection_config()
        if context_window is None:
            context_window = int(cfg.get("context_window", 10))
        self.context_window = context_window

        # Compile patterns: negation first, then affirmative.
        self._patterns: list[_CompiledPattern] = []
        for pat_str, grp, canon in _NEGATION_PATTERNS:
            self._patterns.append(
                _CompiledPattern(
                    re.compile(pat_str, re.IGNORECASE),
                    grp,
                    canon,
                    is_negation=True,
                )
            )
        for pat_str, grp, canon in _AFFIRMATIVE_PATTERNS:
            self._patterns.append(
                _CompiledPattern(
                    re.compile(pat_str, re.IGNORECASE),
                    grp,
                    canon,
                    is_negation=False,
                )
            )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[MatchRecord]:
        """Find all deontic expressions in *normalised* text.

        Returns matches sorted by ``start_char``.  Overlapping spans are
        resolved greedily (first-come wins; negation patterns are
        checked before affirmative ones).
        """
        if not text:
            return []

        # Collect all candidate spans.
        candidates: list[tuple[int, int, _CompiledPattern]] = []
        for cp in self._patterns:
            for m in cp.regex.finditer(text):
                candidates.append((m.start(), m.end(), cp))

        if not candidates:
            return []

        # Sort by start position, then prefer longer match (= earlier negation).
        candidates.sort(key=lambda c: (c[0], -c[1]))

        # Greedy non-overlapping selection.
        selected: list[tuple[int, int, _CompiledPattern]] = []
        last_end = -1
        for start, end, cp in candidates:
            if start >= last_end:
                selected.append((start, end, cp))
                last_end = end

        # Build MatchRecord objects with word-level context.
        matches: list[MatchRecord] = []
        for start, end, cp in selected:
            ctx_before = self._context_before(text, start)
            ctx_after = self._context_after(text, end)
            matches.append(
                MatchRecord(
                    pmid="",  # caller fills this in
                    section="",
                    expression=cp.canonical,
                    expression_group=cp.group,
                    is_negation=cp.is_negation,
                    start_char=start,
                    end_char=end,
                    context_before=ctx_before,
                    context_after=ctx_after,
                )
            )
        return matches

    # ------------------------------------------------------------------
    # Section-level analysis
    # ------------------------------------------------------------------

    def analyze_section(
        self,
        pmid: str,
        section: str,
        raw_text: str,
    ) -> tuple[SectionInventoryRecord, list[MatchRecord]]:
        """Normalise text, detect expressions, and return inventory + matches.

        Parameters
        ----------
        pmid : str
            PubMed ID.
        section : str
            Section name (e.g. ``'abstract_pubmed'``).
        raw_text : str
            Raw (un-normalised) text from the corpus.

        Returns
        -------
        inventory : SectionInventoryRecord
            Summary row with word count and per-group tallies.
        matches : list[MatchRecord]
            Individual match records (may be empty).
        """
        normalised = normalize_for_detection(raw_text) if raw_text else ""
        wc = count_words(normalised)
        raw_matches = self.detect(normalised)

        # Fill in pmid / section for each match.
        for m in raw_matches:
            m.pmid = pmid
            m.section = section

        # Tally per-group counts.
        group_aff: dict[str, int] = {g: 0 for g in EXPRESSION_GROUPS}
        group_neg: dict[str, int] = {g: 0 for g in EXPRESSION_GROUPS}
        for m in raw_matches:
            if m.is_negation:
                group_neg[m.expression_group] = group_neg.get(m.expression_group, 0) + 1
            else:
                group_aff[m.expression_group] = group_aff.get(m.expression_group, 0) + 1

        aff_total = sum(group_aff.values())
        neg_total = sum(group_neg.values())
        total = aff_total + neg_total

        inventory = SectionInventoryRecord(
            pmid=pmid,
            section=section,
            word_count=wc,
            total_deontic=total,
            affirmative_count=aff_total,
            negation_count=neg_total,
            must=group_aff.get("must", 0),
            should=group_aff.get("should", 0),
            need_to=group_aff.get("need_to", 0),
            have_to=group_aff.get("have_to", 0),
            required_to=group_aff.get("required_to", 0),
            must_neg=group_neg.get("must", 0),
            should_neg=group_neg.get("should", 0),
            need_to_neg=group_neg.get("need_to", 0),
            have_to_neg=group_neg.get("have_to", 0),
            required_to_neg=group_neg.get("required_to", 0),
            rate_per_1000=(total / wc * 1000) if wc > 0 else 0.0,
        )

        return inventory, raw_matches

    # ------------------------------------------------------------------
    # Context extraction helpers
    # ------------------------------------------------------------------

    def _context_before(self, text: str, start: int) -> str:
        """Extract up to ``context_window`` words before *start*."""
        prefix = text[:start].rstrip()
        words = prefix.split()
        return " ".join(words[-self.context_window :])

    def _context_after(self, text: str, end: int) -> str:
        """Extract up to ``context_window`` words after *end*."""
        suffix = text[end:].lstrip()
        words = suffix.split()
        return " ".join(words[: self.context_window])


# ---------------------------------------------------------------------------
# Batch detection on corpus
# ---------------------------------------------------------------------------

def run_detection_on_corpus(
    parquet_path: Path,
    inventory_path: Path,
    matches_path: Path,
    sections: list[str] | None = None,
    resume: bool = True,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """Run deontic detection on the entire corpus.

    Parameters
    ----------
    parquet_path : Path
        Input corpus Parquet (``articles_corpus.parquet``).
    inventory_path : Path
        Output ``section_inventory.parquet``.
    matches_path : Path
        Output ``matches.parquet``.
    sections : list[str] | None
        Columns to analyse.  Defaults to ``['abstract_pubmed']``.
    resume : bool
        If True, skip (pmid, section) pairs already in *inventory_path*.
    limit : int | None
        Process at most *limit* articles (for dry-run / debugging).

    Returns
    -------
    (n_articles_processed, n_total_matches)
    """
    if sections is None:
        cfg = get_detection_config()
        sections = cfg.get("default_sections", ["abstract_pubmed"])

    df = pl.read_parquet(parquet_path)
    logger.info("Loaded corpus: %d articles", len(df))

    # Determine which (pmid, section) pairs to skip.
    done_keys: set[tuple[str, str]] = set()
    if resume and inventory_path.exists():
        existing = pl.read_parquet(inventory_path)
        for row in existing.iter_rows(named=True):
            done_keys.add((row["pmid"], row["section"]))
        logger.info("Resume: %d (pmid, section) pairs already done", len(done_keys))

    detector = DeonticDetector()

    inv_rows: list[dict] = []
    match_rows: list[dict] = []
    n_processed = 0
    n_matches = 0
    flush_interval = 100

    pmids = df["pmid"].to_list()
    if limit is not None:
        pmids = pmids[:limit]

    for i, pmid in enumerate(pmids):
        row_data = df.filter(pl.col("pmid") == pmid).to_dicts()[0]

        for sec in sections:
            if (pmid, sec) in done_keys:
                continue
            raw_text = row_data.get(sec) or ""
            inv, matches = detector.analyze_section(pmid, sec, raw_text)
            inv_rows.append(inv.model_dump())
            for m in matches:
                match_rows.append(m.model_dump())
            n_matches += len(matches)

        n_processed += 1

        # Incremental flush.
        if n_processed % flush_interval == 0:
            _flush(inv_rows, match_rows, inventory_path, matches_path)
            inv_rows.clear()
            match_rows.clear()
            logger.info("Flushed after %d articles (%d matches so far)", n_processed, n_matches)

    # Final flush.
    if inv_rows:
        _flush(inv_rows, match_rows, inventory_path, matches_path)

    logger.info(
        "Detection complete: %d articles, %d matches", n_processed, n_matches
    )
    return n_processed, n_matches


def _flush(
    inv_rows: list[dict],
    match_rows: list[dict],
    inventory_path: Path,
    matches_path: Path,
) -> None:
    """Append buffered rows to Parquet files."""
    if inv_rows:
        new_inv = pl.DataFrame(inv_rows)
        if inventory_path.exists():
            # memory_map=False so the existing file is not left mmapped while we
            # overwrite the same path (Windows os error 1224).
            existing = pl.read_parquet(inventory_path, memory_map=False)
            combined = pl.concat([existing, new_inv])
        else:
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_inv
        combined.write_parquet(inventory_path, compression="zstd")

    if match_rows:
        new_matches = pl.DataFrame(match_rows)
        if matches_path.exists():
            existing = pl.read_parquet(matches_path, memory_map=False)
            combined = pl.concat([existing, new_matches])
        else:
            matches_path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_matches
        combined.write_parquet(matches_path, compression="zstd")
    elif not matches_path.exists() and inv_rows:
        # Ensure matches file exists even with zero matches.
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame(
            schema={
                "pmid": pl.Utf8,
                "section": pl.Utf8,
                "expression": pl.Utf8,
                "expression_group": pl.Utf8,
                "is_negation": pl.Boolean,
                "start_char": pl.Int64,
                "end_char": pl.Int64,
                "context_before": pl.Utf8,
                "context_after": pl.Utf8,
            }
        )
        empty.write_parquet(matches_path, compression="zstd")
