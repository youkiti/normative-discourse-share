"""Abstract screening with Gemini API.

Classifies each article as original empirical research (include) or
non-original (exclude) based on title + PubMed abstract.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ScreeningResult(BaseModel):
    """Decision for a single article."""

    pmid: str
    decision: str = Field(description="include | exclude")
    confidence: str = Field(description="high | medium | low")
    reason: str = Field(default="", description="One-sentence justification")
    article_type: str = Field(default="", description="Detected article type")
    status: str = Field(default="ok", description="ok | error | skipped")
    error_detail: str = Field(default="")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SCREENING_PROMPT = """\
You are a screening assistant for a bibliometric study of deontic modal \
expressions in medical literature. Your task is to classify whether an \
article is an ORIGINAL EMPIRICAL RESEARCH article based on its title and \
abstract.

## Definition of "Original Empirical Research"

An article is INCLUDED if it reports PRIMARY data collected and analysed by \
the authors themselves. This includes:
- Randomised controlled trials (RCTs)
- Observational studies (cohort, case-control, cross-sectional)
- Qualitative studies (interviews, focus groups, ethnography, discourse analysis)
- Mixed-methods studies
- Survey-based studies
- Experimental or quasi-experimental studies
- Secondary analyses of existing datasets where the authors perform novel \
statistical analysis on individual-level data
- Diagnostic / prognostic accuracy studies
- Cost-effectiveness analyses with original modelling
- Simulation / modelling studies that use original computational models with \
novel parameter estimation

An article is EXCLUDED if it does NOT report original empirical data. Examples:
- Systematic reviews, scoping reviews, meta-analyses, narrative reviews
- Editorials, commentaries, viewpoints, perspectives, opinion pieces
- Tutorials, how-to guides, "tips" articles, best-practice guidelines
- Conceptual / theoretical papers without empirical data collection
- Methodological guidance or framework papers (e.g., GRADE guides, reporting \
guidelines) that do not collect or analyse primary data
- Consensus statements, position papers, policy analyses
- Letters to the editor, correspondence, rebuttals
- Case reports or case series with fewer than 5 cases
- News items, book reviews, historical essays
- Innovation reports that describe a programme or tool WITHOUT evaluation data

## Important Edge Cases

- Programme descriptions that include evaluation with outcome data → INCLUDE
- Programme descriptions without formal evaluation → EXCLUDE
- "Lessons learned" or reflective accounts without systematic data collection \
→ EXCLUDE

## Input

Title: {title}
Abstract: {abstract}

## Output

Respond with a JSON object with these exact keys:
- "decision": "include" or "exclude"
- "confidence": "high", "medium", or "low"
- "reason": one-sentence justification
- "article_type": detected article type (e.g. "RCT", "qualitative study", \
"systematic review", "editorial", "conceptual paper", etc.)
"""


# ---------------------------------------------------------------------------
# Gemini client helpers
# ---------------------------------------------------------------------------

def _build_client(api_key: str) -> Any:
    """Create a google-genai client."""
    from google import genai

    return genai.Client(api_key=api_key)


def _parse_response(raw_text: str, pmid: str) -> ScreeningResult:
    """Parse Gemini JSON response into ScreeningResult."""
    try:
        data = json.loads(raw_text)
        return ScreeningResult(
            pmid=pmid,
            decision=data.get("decision", "exclude"),
            confidence=data.get("confidence", "low"),
            reason=data.get("reason", ""),
            article_type=data.get("article_type", ""),
            status="ok",
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return ScreeningResult(
            pmid=pmid,
            decision="exclude",
            confidence="low",
            reason="",
            article_type="",
            status="error",
            error_detail=f"Parse error: {exc}; raw={raw_text[:200]}",
        )


# ---------------------------------------------------------------------------
# Async screening
# ---------------------------------------------------------------------------

MODEL_ID = "gemini-3.1-flash-lite-preview"


async def _screen_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    pmid: str,
    title: str,
    abstract: str,
) -> ScreeningResult:
    """Screen a single article (async, respects semaphore)."""
    from google.genai import types

    prompt = SCREENING_PROMPT.format(title=title, abstract=abstract or "(no abstract)")

    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_ID,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                    response_mime_type="application/json",
                ),
            )
            raw = response.text or ""
            return _parse_response(raw, pmid)
        except Exception as exc:
            logger.warning("Gemini API error for PMID %s: %s", pmid, exc)
            return ScreeningResult(
                pmid=pmid,
                decision="exclude",
                confidence="low",
                status="error",
                error_detail=str(exc)[:300],
            )


async def _run_async(
    client: Any,
    articles: list[dict[str, str]],
    writer: _ManifestWriter,
    concurrency: int = 20,
    batch_size: int = 20,
) -> list[ScreeningResult]:
    """Run screening for all articles with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[ScreeningResult] = []
    total = len(articles)
    n_batches = math.ceil(total / batch_size)

    for batch_idx in range(n_batches):
        batch = articles[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        tasks = [
            _screen_one(client, semaphore, a["pmid"], a["title"], a["abstract"])
            for a in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        for r in batch_results:
            results.append(r)
            writer.write_row(r)
        done = min((batch_idx + 1) * batch_size, total)
        logger.info(
            "Screening progress: %d/%d (batch %d/%d)",
            done, total, batch_idx + 1, n_batches,
        )

    return results


# ---------------------------------------------------------------------------
# Manifest writer (incremental CSV persistence)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "pmid", "decision", "confidence", "reason", "article_type",
    "status", "error_detail",
]


class _ManifestWriter:
    """Append screening results to CSV one row at a time."""

    def __init__(self, path: Path, append: bool = False) -> None:
        self._path = path
        mode = "a" if append else "w"
        self._fh = open(path, mode, newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_COLUMNS)
        if not append:
            self._writer.writeheader()

    def write_row(self, result: ScreeningResult) -> None:
        self._writer.writerow(result.model_dump(include=set(_CSV_COLUMNS)))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch_screening(
    input_parquet: Path,
    output_csv: Path,
    api_key: str,
    *,
    resume: bool = True,
    concurrency: int = 20,
    limit: int | None = None,
) -> pd.DataFrame:
    """Screen all articles in the corpus Parquet.

    Parameters
    ----------
    input_parquet : Path to articles_corpus.parquet
    output_csv : Path for screening results CSV
    api_key : Gemini API key
    resume : If True, skip PMIDs already in output_csv
    concurrency : Max concurrent Gemini API calls
    limit : If set, only process this many articles (for dry-run)

    Returns
    -------
    pd.DataFrame with screening results
    """
    import polars as pl

    # Load corpus (polars only — no pyarrow/pandas conversion)
    pldf = pl.read_parquet(input_parquet).select(
        ["pmid", "title", "abstract_pubmed"]
    )
    logger.info("Loaded %d articles from %s", len(pldf), input_parquet)

    # Build article list
    articles = [
        {
            "pmid": str(row[0] or ""),
            "title": str(row[1] or ""),
            "abstract": str(row[2] or ""),
        }
        for row in pldf.iter_rows()
    ]

    # Resume: skip already-processed PMIDs
    done_pmids: set[str] = set()
    append_mode = False
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv, dtype=str)
        done_pmids = set(existing["pmid"].tolist())
        logger.info("Resume: %d PMIDs already processed", len(done_pmids))
        append_mode = True

    pending = [a for a in articles if a["pmid"] not in done_pmids]

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        logger.info("No articles to screen (all done or empty input)")
        if output_csv.exists():
            return pd.read_csv(output_csv, dtype=str)
        return pd.DataFrame(columns=_CSV_COLUMNS)

    logger.info("Screening %d articles (concurrency=%d)", len(pending), concurrency)

    # Build client
    client = _build_client(api_key)

    # Write results
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = _ManifestWriter(output_csv, append=append_mode)

    try:
        results = asyncio.run(
            _run_async(client, pending, writer, concurrency=concurrency)
        )
    finally:
        writer.close()

    # Summarise
    ok = sum(1 for r in results if r.status == "ok")
    errors = sum(1 for r in results if r.status == "error")
    included = sum(1 for r in results if r.decision == "include" and r.status == "ok")
    excluded = sum(1 for r in results if r.decision == "exclude" and r.status == "ok")
    logger.info(
        "Screening complete: %d ok (%d include, %d exclude), %d errors",
        ok, included, excluded, errors,
    )

    return pd.read_csv(output_csv, dtype=str)


def generate_review_sample(
    screening_csv: Path,
    output_tsv: Path,
    corpus_parquet: Path,
    *,
    seed: int = 20260309,
) -> pd.DataFrame:
    """Generate a stratified sample for human review.

    Sampling rates:
    - confidence=low → 100%
    - confidence=medium → 20%
    - confidence=high + exclude → 10%
    - confidence=high + include → 5%
    """
    import polars as pl

    rng = random.Random(seed)

    screening = pd.read_csv(screening_csv, dtype=str)
    screening = screening[screening["status"] == "ok"].copy()

    # Load title/journal from corpus for context (polars → dict → pandas merge)
    corpus_pl = pl.read_parquet(corpus_parquet).select(
        ["pmid", "title", "journal", "abstract_pubmed"]
    )
    corpus_records = [
        {"pmid": str(r[0]), "title": r[1], "journal": r[2], "abstract_pubmed": r[3]}
        for r in corpus_pl.iter_rows()
    ]
    corpus = pd.DataFrame(corpus_records)
    merged = screening.merge(corpus, on="pmid", how="left")

    # Stratified sampling
    sample_indices: list[int] = []

    for idx, row in merged.iterrows():
        conf = row.get("confidence", "low")
        dec = row.get("decision", "exclude")

        if conf == "low":
            sample_indices.append(idx)
        elif conf == "medium":
            if rng.random() < 0.20:
                sample_indices.append(idx)
        elif conf == "high" and dec == "exclude":
            if rng.random() < 0.10:
                sample_indices.append(idx)
        elif conf == "high" and dec == "include":
            if rng.random() < 0.05:
                sample_indices.append(idx)

    sample = merged.loc[sample_indices].copy()

    # Add review columns
    sample["review_decision"] = ""  # agree / disagree
    sample["review_corrected_decision"] = ""  # include / exclude (if disagree)
    sample["review_notes"] = ""

    # Reorder columns
    col_order = [
        "pmid", "title", "journal",
        "decision", "confidence", "reason", "article_type",
        "abstract_pubmed",
        "review_decision", "review_corrected_decision", "review_notes",
    ]
    sample = sample[[c for c in col_order if c in sample.columns]]

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")

    # Summary by stratum
    low_n = sum(1 for _, r in screening.iterrows() if r["confidence"] == "low")
    med_n = sum(1 for _, r in screening.iterrows() if r["confidence"] == "medium")
    high_exc_n = sum(
        1 for _, r in screening.iterrows()
        if r["confidence"] == "high" and r["decision"] == "exclude"
    )
    high_inc_n = sum(
        1 for _, r in screening.iterrows()
        if r["confidence"] == "high" and r["decision"] == "include"
    )

    logger.info(
        "Review sample: %d articles (low=%d/%d, medium=%d/%d, "
        "high+exclude=%d/%d, high+include=%d/%d)",
        len(sample),
        sum(1 for _, r in sample.iterrows() if r["confidence"] == "low"), low_n,
        sum(1 for _, r in sample.iterrows() if r["confidence"] == "medium"), med_n,
        sum(
            1 for _, r in sample.iterrows()
            if r["confidence"] == "high" and r["decision"] == "exclude"
        ), high_exc_n,
        sum(
            1 for _, r in sample.iterrows()
            if r["confidence"] == "high" and r["decision"] == "include"
        ), high_inc_n,
    )
    logger.info("Saved review sample: %s", output_tsv)

    return sample
