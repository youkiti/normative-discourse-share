"""Frequency analysis and statistical tests for deontic expression detection.

Provides:
- Descriptive statistics (corpus-level and article-level)
- G-squared (log-likelihood ratio) test for field comparison
- Rate ratio with 95% CI (log-normal approximation) — primary effect measure
- Bonferroni correction for multiple comparisons

The reported effect measure follows ``doc/analysis_plan_current.md``: per field
count and rate per 1,000 words, the rate ratio with 95% CI, the G-squared
statistic, and the Bonferroni-corrected p-value.  No other effect measures
(log₂ ratio, rate difference, Cramér's V) are reported.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field

from normative_discourse.config.settings import get_statistics_config
from normative_discourse.schemas import EXPRESSION_GROUPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class GTestResult(BaseModel):
    """Result of a G-squared (log-likelihood ratio) test for one expression group."""

    expression_group: str
    count_group1: int = Field(description="Count in group 1 (general_medicine)")
    count_group2: int = Field(description="Count in group 2 (medical_education)")
    words_group1: int
    words_group2: int
    rate_group1: float = Field(description="Per 1,000 words")
    rate_group2: float = Field(description="Per 1,000 words")
    g_squared: float
    df: int = Field(default=1)
    p_value: float
    p_value_corrected: float = Field(description="After Bonferroni correction")
    significant: bool
    rate_ratio: float = Field(description="ME / GM (reference: general_medicine) — PRIMARY effect measure")
    rate_ratio_ci_lower: float = Field(description="95% CI lower for rate ratio")
    rate_ratio_ci_upper: float = Field(description="95% CI upper for rate ratio")


class FieldDescriptive(BaseModel):
    """Descriptive statistics for one field or journal."""

    group_name: str = Field(description="Journal abbreviation or field name")
    field: str
    n_articles: int
    total_words: int
    total_deontic: int
    rate_per_1000: float
    rate_per_million: float
    article_rate_mean: float
    article_rate_sd: float
    article_rate_median: float
    article_rate_q1: float
    article_rate_q3: float
    article_rate_min: float
    article_rate_max: float


class FrequencyReport(BaseModel):
    """Complete frequency analysis report."""

    analysis_section: str
    total_articles: int
    total_words: int
    total_deontic: int
    descriptive_by_field: list[FieldDescriptive]
    descriptive_by_journal: list[FieldDescriptive]
    g_test_results: list[GTestResult]
    alpha: float
    correction_method: str
    n_comparisons: int


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def compute_g_squared(
    count_1: int,
    words_1: int,
    count_2: int,
    words_2: int,
) -> tuple[float, float]:
    """Compute G-squared statistic and p-value for a 2×2 table.

    Table layout:
        |         | target  | non-target |
        |---------|---------|------------|
        | group1  |   O11   |    O12     |
        | group2  |   O21   |    O22     |

    G² = 2 Σ O_ij ln(O_ij / E_ij)   (0 · ln(0) = 0 by convention)

    Returns (g_squared, p_value).
    """
    from scipy.stats import chi2

    o11 = count_1
    o12 = words_1 - count_1
    o21 = count_2
    o22 = words_2 - count_2

    n = o11 + o12 + o21 + o22
    if n == 0:
        return 0.0, 1.0

    row_totals = [o11 + o12, o21 + o22]
    col_totals = [o11 + o21, o12 + o22]

    g2 = 0.0
    for i, row in enumerate([[o11, o12], [o21, o22]]):
        for j, obs in enumerate(row):
            expected = row_totals[i] * col_totals[j] / n
            if obs > 0 and expected > 0:
                g2 += obs * math.log(obs / expected)
    g2 *= 2

    p = float(chi2.sf(g2, df=1))
    return g2, p


def compute_rate_ratio_ci(
    count_1: int,
    words_1: int,
    count_2: int,
    words_2: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute rate ratio and 95% CI (log-normal approximation).

    RR = (count_1/words_1) / (count_2/words_2)
    SE(ln RR) = sqrt(1/count_1 + 1/count_2)
    CI: exp(ln(RR) ± z * SE)

    Returns (rate_ratio, ci_lower, ci_upper).
    """
    from scipy.stats import norm

    if count_1 == 0 or count_2 == 0 or words_1 == 0 or words_2 == 0:
        rate_1 = count_1 / words_1 if words_1 > 0 else 0.0
        rate_2 = count_2 / words_2 if words_2 > 0 else 0.0
        rr = rate_1 / rate_2 if rate_2 > 0 else 0.0
        return rr, 0.0, float("inf")

    rate_1 = count_1 / words_1
    rate_2 = count_2 / words_2
    rr = rate_1 / rate_2

    ln_rr = math.log(rr)
    se = math.sqrt(1.0 / count_1 + 1.0 / count_2)
    z = float(norm.ppf(1 - alpha / 2))

    ci_lower = math.exp(ln_rr - z * se)
    ci_upper = math.exp(ln_rr + z * se)
    return rr, ci_lower, ci_upper


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply Bonferroni correction.

    Returns list of (p_corrected, is_significant).
    """
    n = len(p_values)
    if n == 0:
        return []
    return [(min(p * n, 1.0), min(p * n, 1.0) < alpha) for p in p_values]


# ---------------------------------------------------------------------------
# Descriptive statistics helpers
# ---------------------------------------------------------------------------

def _compute_descriptive(
    inv_df: pl.DataFrame,
    group_col: str,
    group_val: str,
    field_val: str,
) -> FieldDescriptive:
    """Compute descriptive statistics for a subset of the inventory."""
    subset = inv_df.filter(pl.col(group_col) == group_val)
    n = len(subset)
    total_words = int(subset["word_count"].sum())
    total_deontic = int(subset["total_deontic"].sum())

    rate_1000 = total_deontic / total_words * 1000 if total_words > 0 else 0.0
    rate_million = total_deontic / total_words * 1_000_000 if total_words > 0 else 0.0

    # Article-level rate distribution.
    article_rates = subset.with_columns(
        (pl.col("total_deontic") / pl.col("word_count") * 1000)
        .alias("rate")
    )["rate"]

    if n > 0:
        mean_rate = float(article_rates.mean())
        sd_rate = float(article_rates.std()) if n > 1 else 0.0
        median_rate = float(article_rates.median())
        q1 = float(article_rates.quantile(0.25))
        q3 = float(article_rates.quantile(0.75))
        min_rate = float(article_rates.min())
        max_rate = float(article_rates.max())
    else:
        mean_rate = sd_rate = median_rate = q1 = q3 = min_rate = max_rate = 0.0

    return FieldDescriptive(
        group_name=group_val,
        field=field_val,
        n_articles=n,
        total_words=total_words,
        total_deontic=total_deontic,
        rate_per_1000=rate_1000,
        rate_per_million=rate_million,
        article_rate_mean=mean_rate,
        article_rate_sd=sd_rate,
        article_rate_median=median_rate,
        article_rate_q1=q1,
        article_rate_q3=q3,
        article_rate_min=min_rate,
        article_rate_max=max_rate,
    )


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def run_frequency_analysis(
    inventory_path: Path,
    corpus_path: Path,
    output_dir: Path,
    section: str = "abstract_pubmed",
    alpha: Optional[float] = None,
    correction_method: Optional[str] = None,
    sections: Optional[list[str]] = None,
    imrad_only: bool = False,
) -> FrequencyReport:
    """Run complete frequency analysis on detection results.

    Reads section_inventory.parquet (for counts) and corpus Parquet
    (for journal/field metadata).  Writes CSV tables and JSON report.

    Parameters
    ----------
    section : str
        Label for this analysis (also the default single section to analyse
        and the value stored in ``analysis_section``).
    sections : list[str] | None
        Sections to include.  When more than one is given, per-article tallies
        are summed across those sections so the analysis unit is the combined
        text (e.g. abstract + IMRaD body).  Defaults to ``[section]``.
    imrad_only : bool
        If True, non-IMRaD records (``is_non_imrad``) are excluded before any
        counting, so the denominator is the IMRaD subset only.
    """
    stats_cfg = get_statistics_config()
    if alpha is None:
        alpha = float(stats_cfg.get("alpha", 0.05))
    if correction_method is None:
        correction_method = stats_cfg.get("correction_method", "bonferroni")

    # Resolve which sections make up the analysis unit.
    sec_list = sections if sections else [section]

    # Load data.
    inv_df = pl.read_parquet(inventory_path, memory_map=False).filter(
        pl.col("section").is_in(sec_list)
    )

    # When combining multiple sections (e.g. abstract + IMRaD body), sum the
    # per-section tallies per article so the analysis unit is the combined text.
    if len(sec_list) > 1:
        tally_cols = ["word_count", "total_deontic", "affirmative_count", "negation_count"]
        for grp in EXPRESSION_GROUPS:
            tally_cols.append(grp)
            if f"{grp}_neg" in inv_df.columns:
                tally_cols.append(f"{grp}_neg")
        tally_cols = [c for c in tally_cols if c in inv_df.columns]
        inv_df = (
            inv_df.group_by("pmid")
            .agg([pl.col(c).sum() for c in tally_cols])
            .with_columns(
                pl.when(pl.col("word_count") > 0)
                .then(pl.col("total_deontic") / pl.col("word_count") * 1000)
                .otherwise(0.0)
                .alias("rate_per_1000")
            )
        )

    # Corpus metadata (+ is_non_imrad when restricting to the IMRaD subset).
    meta_cols = ["pmid", "journal", "journal_field"]
    if imrad_only:
        meta_cols.append("is_non_imrad")
    corpus_df = pl.read_parquet(corpus_path, memory_map=False).select(meta_cols)

    # Join metadata.
    inv_df = inv_df.join(corpus_df, on="pmid", how="left")

    # IMRaD restriction: drop non-IMRaD records entirely (essays/letters/broken
    # extracts have no IMRaD boundary and are excluded from the modal analysis).
    n_nonimrad = 0
    if imrad_only:
        n_pre = len(inv_df)
        inv_df = inv_df.filter(~pl.col("is_non_imrad").fill_null(False))
        n_nonimrad = n_pre - len(inv_df)
        inv_df = inv_df.drop("is_non_imrad")

    # Section-level denominator policy: restrict to the non-empty subset of
    # records (word_count > 0).  Empty sections carry no text and would
    # otherwise dilute the per-article rate descriptives with spurious zeros
    # without changing the corpus-level rate.
    n_before = len(inv_df)
    inv_df = inv_df.filter(pl.col("word_count") > 0)
    n_empty = n_before - len(inv_df)

    logger.info(
        "Analysis '%s' over sections %s: %d articles "
        "(%d non-IMRaD excluded, %d empty-section rows excluded)",
        section,
        sec_list,
        len(inv_df),
        n_nonimrad,
        n_empty,
    )

    total_articles = len(inv_df)
    total_words = int(inv_df["word_count"].sum())
    total_deontic = int(inv_df["total_deontic"].sum())

    # ------------------------------------------------------------------
    # Descriptive by field
    # ------------------------------------------------------------------
    fields = ["general_medicine", "medical_education"]
    desc_by_field = []
    for f in fields:
        desc_by_field.append(_compute_descriptive(inv_df, "journal_field", f, f))

    # ------------------------------------------------------------------
    # Descriptive by journal
    # ------------------------------------------------------------------
    journals = sorted(inv_df["journal"].unique().to_list())
    desc_by_journal = []
    for j in journals:
        row = inv_df.filter(pl.col("journal") == j)
        field_val = row["journal_field"][0] if len(row) > 0 else ""
        desc_by_journal.append(_compute_descriptive(inv_df, "journal", j, field_val))

    # ------------------------------------------------------------------
    # G² tests per expression group (GM vs ME)
    # ------------------------------------------------------------------
    gm = inv_df.filter(pl.col("journal_field") == "general_medicine")
    me = inv_df.filter(pl.col("journal_field") == "medical_education")
    words_gm = int(gm["word_count"].sum())
    words_me = int(me["word_count"].sum())

    g_results: list[GTestResult] = []
    raw_p: list[float] = []

    for grp in EXPRESSION_GROUPS:
        # Sum affirmative + negation for each group.
        neg_col = f"{grp}_neg" if f"{grp}_neg" in inv_df.columns else None
        count_gm = int(gm[grp].sum())
        count_me = int(me[grp].sum())
        if neg_col:
            count_gm += int(gm[neg_col].sum())
            count_me += int(me[neg_col].sum())

        rate_gm = count_gm / words_gm * 1000 if words_gm > 0 else 0.0
        rate_me = count_me / words_me * 1000 if words_me > 0 else 0.0

        g2, p = compute_g_squared(count_gm, words_gm, count_me, words_me)
        # Primary effect measure is the rate ratio, oriented as medical_education
        # relative to general_medicine (ME / GM, reference: GM).
        rr, ci_lo, ci_hi = compute_rate_ratio_ci(count_me, words_me, count_gm, words_gm, alpha)

        raw_p.append(p)
        g_results.append(
            GTestResult(
                expression_group=grp,
                count_group1=count_gm,
                count_group2=count_me,
                words_group1=words_gm,
                words_group2=words_me,
                rate_group1=rate_gm,
                rate_group2=rate_me,
                g_squared=g2,
                p_value=p,
                p_value_corrected=0.0,  # filled below
                significant=False,
                rate_ratio=rr,
                rate_ratio_ci_lower=ci_lo,
                rate_ratio_ci_upper=ci_hi,
            )
        )

    # Apply Bonferroni correction.
    corrected = bonferroni_correct(raw_p, alpha)
    for i, (p_corr, sig) in enumerate(corrected):
        g_results[i].p_value_corrected = p_corr
        g_results[i].significant = sig

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    report = FrequencyReport(
        analysis_section=section,
        total_articles=total_articles,
        total_words=total_words,
        total_deontic=total_deontic,
        descriptive_by_field=desc_by_field,
        descriptive_by_journal=desc_by_journal,
        g_test_results=g_results,
        alpha=alpha,
        correction_method=correction_method,
        n_comparisons=len(EXPRESSION_GROUPS),
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # article_expression_counts.csv
    cols = ["pmid", "journal", "journal_field", "word_count"]
    for grp in EXPRESSION_GROUPS:
        cols.append(grp)
        neg_col = f"{grp}_neg"
        if neg_col in inv_df.columns:
            cols.append(neg_col)
    cols.extend(["total_deontic", "rate_per_1000"])
    inv_df.select([c for c in cols if c in inv_df.columns]).write_csv(
        output_dir / "article_expression_counts.csv"
    )

    # descriptive_by_field.csv
    _write_descriptive_csv(desc_by_field, output_dir / "descriptive_by_field.csv")

    # descriptive_by_journal.csv
    _write_descriptive_csv(desc_by_journal, output_dir / "descriptive_by_journal.csv")

    # g_test_results.csv
    g_rows = [r.model_dump() for r in g_results]
    pl.DataFrame(g_rows).write_csv(output_dir / "g_test_results.csv")

    # frequency_report.json
    with open(output_dir / "frequency_report.json", "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info("Wrote frequency analysis results to %s", output_dir)
    return report


def _write_descriptive_csv(
    descriptives: list[FieldDescriptive],
    path: Path,
) -> None:
    """Write descriptive statistics to CSV."""
    rows = [d.model_dump() for d in descriptives]
    pl.DataFrame(rows).write_csv(path)
