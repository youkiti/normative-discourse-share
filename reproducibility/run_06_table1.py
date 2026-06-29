"""Step 06 -- reproduce Table 1 (article characteristics).

EN: Computes the Table 1 numbers DIRECTLY from the frozen corpus, restricted to
    the analysis set `~is_non_imrad` (N = 10,743). This deliberately does NOT use
    `describe-corpus`, which reports the full 10,899-row corpus (incl. 156
    non-IMRaD) and so would not match the manuscript.
JA: Table 1 を frozen コーパスから直接算出。解析対象 `~is_non_imrad`（N=10,743）に限定する。
    describe-corpus は非 IMRaD 156 を含む全 10,899 行を出すため Table 1 とは一致しない。

Input   : data/processed/articles_corpus_10yr_clean.parquet
Outputs : reproducibility/work/table1_counts.csv     (journal / field / period / presence)
          reproducibility/work/table1_section_words.csv (section length median + IQR)

    python reproducibility/run_06_table1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The final summary prints a polars table whose box-drawing characters cannot be
# encoded by legacy consoles (e.g. cp932 on Japanese Windows), which would crash
# this step AFTER the CSVs are already written. Force UTF-8 on our own stdout so
# the documented standalone invocation works regardless of the console codepage.
# (run_all.py already sets PYTHONUTF8=1 for child processes; this covers the
# direct `python run_06_table1.py` case.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # stdout may not be a reconfigurable stream
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from _env import CLEAN_PARQUET  # noqa: E402

WORK = Path(__file__).resolve().parent / "work"
SECTION_COLS = ["abstract_pubmed", "introduction", "methods", "results", "discussion"]
JOURNAL_ORDER = {
    "medical_education": ["Acad Med", "Med Teach", "Med Educ", "JMIR Med Educ", "Perspect Med Educ"],
    "general_medicine": ["N Engl J Med", "Lancet", "JAMA", "BMJ", "Ann Intern Med"],
}


def _present(df: pl.DataFrame, col: str) -> int:
    return df.filter(pl.col(col).is_not_null() & (pl.col(col).str.strip_chars() != "")).height


def _wordcount_expr(col: str) -> pl.Expr:
    # Whitespace tokenisation, matching the corpus describer's notion of "words".
    return (
        pl.col(col).fill_null("").str.strip_chars().str.split(by=" ")
        .list.eval(pl.element().filter(pl.element().str.strip_chars() != ""))
        .list.len()
        .alias(f"_wc_{col}")
    )


def main() -> int:
    if not CLEAN_PARQUET.exists():
        print(f"[FAIL] frozen corpus not found: {CLEAN_PARQUET}")
        print("Run run_02_strip_references.py first.")
        return 1

    WORK.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(CLEAN_PARQUET).filter(~pl.col("is_non_imrad"))
    total = df.height
    print(f"Analysis set (~is_non_imrad): N = {total}")

    # --- counts: field, journal, period, section presence ---
    rows: list[dict] = []
    rows.append({"category": "total", "key": "Overall", "field": "all", "n": total})
    for field in ("medical_education", "general_medicine"):
        fdf = df.filter(pl.col("journal_field") == field)
        rows.append({"category": "field_total", "key": field, "field": field, "n": fdf.height})
        for jr in JOURNAL_ORDER[field]:
            n = fdf.filter(pl.col("journal") == jr).height
            pct = 100.0 * n / fdf.height if fdf.height else 0.0
            rows.append({"category": "journal", "key": jr, "field": field, "n": n,
                         "pct_of_field": round(pct, 1)})

    # Period buckets follow the manuscript: the upper bucket is OPEN-ENDED (>=2021),
    # so the 88 articles stamped pub_year=2026 (early-access / ahead-of-print) fold
    # into "2021-2025". Splitting at 2020 keeps the two buckets summing to N.
    pyear = df.with_columns(pl.col("pub_year").cast(pl.Int64))
    period_masks = (
        ("2016-2020", pl.col("pub_year") <= 2020),
        ("2021-2025", pl.col("pub_year") >= 2021),
    )
    for label, mask in period_masks:
        sub = pyear.filter(mask)
        rows.append({"category": "period", "key": label, "field": "all", "n": sub.height})
        for field in ("medical_education", "general_medicine"):
            rows.append({"category": "period", "key": f"{label}/{field}", "field": field,
                         "n": sub.filter(pl.col("journal_field") == field).height})

    for col in ["introduction", "methods", "results", "discussion", "conclusion"]:
        rows.append({"category": "section_present", "key": col, "field": "all",
                     "n": _present(df, col)})

    counts_df = pl.DataFrame(rows)
    counts_path = WORK / "table1_counts.csv"
    counts_df.write_csv(counts_path)

    # --- section length: median + IQR (overall and by field) ---
    wc = df.with_columns([_wordcount_expr(c) for c in SECTION_COLS] + [
        pl.col("word_count").alias("_wc_body")
    ])
    swc_rows: list[dict] = []
    for col in ["_wc_body"] + [f"_wc_{c}" for c in SECTION_COLS]:
        name = col.replace("_wc_", "")
        for field in ("all", "medical_education", "general_medicine"):
            sub = wc if field == "all" else wc.filter(pl.col("journal_field") == field)
            # Section length is computed over articles where the section is present.
            vals = sub.filter(pl.col(col) > 0)[col]
            if vals.len() == 0:
                continue
            swc_rows.append({
                "section": name, "field": field, "n": vals.len(),
                "median": float(vals.median()),
                "q1": float(vals.quantile(0.25)), "q3": float(vals.quantile(0.75)),
            })
    swc_df = pl.DataFrame(swc_rows)
    swc_path = WORK / "table1_section_words.csv"
    swc_df.write_csv(swc_path)

    print(f"\nWrote: {counts_path}")
    print(f"Wrote: {swc_path}")
    print("\n--- key counts ---")
    with pl.Config(tbl_rows=40):
        print(counts_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
