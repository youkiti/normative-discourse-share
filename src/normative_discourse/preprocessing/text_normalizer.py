"""Text normalisation utilities for deontic expression detection.

Handles structured-abstract label stripping, Unicode quote normalisation,
and whitespace collapsing.  All functions are pure (no I/O, no side effects).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Structured abstract labels
# ---------------------------------------------------------------------------
# PubMed structured abstracts embed section labels such as
# "BACKGROUND:", "METHODS:", "CONCLUSIONS AND RELEVANCE:" directly in the
# abstract text.  These labels are PubMed metadata — not author prose — and
# must be removed before word counting or expression detection.
#
# The regex matches labels that appear either at the very start of the text
# or immediately after a sentence-ending period + space boundary.  This
# prevents removing the word "background" when it appears *inside* a
# sentence (e.g. "…against a background of rising costs…").

_ABSTRACT_LABELS = (
    # Standard IMRaD
    r"BACKGROUND",
    r"INTRODUCTION",
    r"OBJECTIVES?",
    r"PURPOSE",
    r"AIMS?",
    r"CONTEXT",
    r"HYPOTHESIS",
    r"DESIGN",
    r"SETTING",
    r"PARTICIPANTS?",
    r"PATIENTS?",
    r"SUBJECTS?",
    r"INTERVENTIONS?",
    r"EXPOSURES?",
    r"METHODS?",
    r"MEASURES?",
    r"MAIN\s+OUTCOMES?\s+(?:AND\s+)?MEASURES?",
    r"OUTCOMES?",
    r"RESULTS?",
    r"FINDINGS?",
    r"CONCLUSIONS?",
    r"INTERPRETATION",
    r"SUMMARY",
    # JAMA extensions
    r"IMPORTANCE",
    r"MEANING",
    r"SIGNIFICANCE",
    r"CONCLUSIONS?\s+AND\s+RELEVANCE",
    # Systematic review / meta-analysis
    r"DATA\s+SOURCES?",
    r"STUDY\s+SELECTION",
    r"DATA\s+EXTRACTION",
    r"DATA\s+SYNTHESIS",
    # Trial registration
    r"TRIAL\s+REGISTRATION",
    r"CLINICAL\s+TRIAL\s+REGISTRATION",
)

# Build a single alternation pattern (longest labels first to avoid partial
# matches).  The label must be followed by optional whitespace + colon + space.
_labels_sorted = sorted(_ABSTRACT_LABELS, key=len, reverse=True)
_LABEL_RE = re.compile(
    r"(?:^|(?<=\.\s))"  # start of string  OR  after ". "
    r"(?:" + "|".join(_labels_sorted) + r")"
    r"\s*:\s*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Quote normalisation
# ---------------------------------------------------------------------------
# Map common curly / smart quote characters to ASCII equivalents so that
# contraction detection (mustn't, shouldn't, …) works uniformly.
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
        "\u201A": "'",   # SINGLE LOW-9 QUOTATION MARK
        "\u02BC": "'",   # MODIFIER LETTER APOSTROPHE
        "\u201C": '"',   # LEFT DOUBLE QUOTATION MARK
        "\u201D": '"',   # RIGHT DOUBLE QUOTATION MARK
        "\u2014": "-",   # EM DASH
        "\u2013": "-",   # EN DASH
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_structured_abstract_labels(text: str) -> str:
    """Remove PubMed structured-abstract section labels.

    Labels like ``BACKGROUND:``, ``METHODS:``, ``CONCLUSIONS AND RELEVANCE:``
    are stripped only when they appear at label positions (start of text or
    after sentence boundaries).  Inline occurrences of the words are left
    intact.
    """
    return _LABEL_RE.sub("", text)


def normalize_quotes(text: str) -> str:
    """Replace curly / smart quotes and dashes with ASCII equivalents."""
    return text.translate(_QUOTE_MAP)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_detection(text: str) -> str:
    """Apply the full normalisation pipeline for deontic detection.

    Order: label stripping → quote normalisation → whitespace collapsing.
    Does **not** lowercase (regex patterns use ``re.IGNORECASE``).
    """
    text = strip_structured_abstract_labels(text)
    text = normalize_quotes(text)
    text = normalize_whitespace(text)
    return text


def count_words(text: str) -> int:
    """Count words via whitespace split.

    Matches the counting method used in ``corpus_builder.py``
    (``len(text.split())``).
    """
    if not text:
        return 0
    return len(text.split())
