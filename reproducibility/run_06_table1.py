"""Step 06 -- reproduce Table 1 (article characteristics).

EN: Computes the Table 1 numbers DIRECTLY from the frozen corpus, restricted to
    the analysis set `~is_non_imrad` (N = 10,743). This deliberately does NOT use
    `describe-corpus`, which reports the full 10,899-row corpus (incl. 156
    non-IMRaD) and so would not match the manuscript.
JA: Table 1 を frozen コーパスから直接算出。解析対象 `~is_non_imrad`（N=10,743）に限定する。
    describe-corpus は非 IMRaD 156 を含む全 10,899 行を出すため Table 1 とは一致しない。

Input   : data/processed/articles_corpus_10yr_clean.parquet
          data/processed/section_inventory.parquet (step 03; section lengths)
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

from _env import CLEAN_PARQUET, INVENTORY_PARQUET  # noqa: E402

WORK = Path(__file__).resolve().parent / "work"
LENGTH_UNITS = {
    "body": ["abstract_pubmed", "introduction", "methods", "results", "discussion", "conclusion"],
    "abstract_pubmed": ["abstract_pubmed"],
    "introduction": ["introduction"],
    "methods": ["methods"],
    "results": ["results"],
    "discussion": ["discussion", "conclusion"],
}
JOURNAL_ORDER = {
    "medical_education": ["Acad Med", "Med Teach", "Med Educ", "JMIR Med Educ", "Perspect Med Educ"],
    "general_medicine": ["N Engl J Med", "Lancet", "JAMA", "BMJ", "Ann Intern Med"],
}


def _present(df: pl.DataFrame, col: str) -> int:
    return df.filter(pl.col(col).is_not_null() & (pl.col(col).str.strip_chars() != "")).height


def main() -> int:
    if not CLEAN_PARQUET.exists():
        print(f"[FAIL] frozen corpus not found: {CLEAN_PARQUET}")
        print("Run run_02_strip_references.py first.")
        return 1
    if not INVENTORY_PARQUET.exists():
        print(f"[FAIL] section inventory not found: {INVENTORY_PARQUET}")
        print("Run run_03_detect_deontic.py first.")
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
    # Lengths use the word counts of the analysed text (section_inventory.parquet
    # from step 03), i.e. the same denominators as the rate models: "body" is the
    # PubMed abstract + Introduction..Conclusion, "abstract_pubmed" is the PubMed
    # abstract, and "discussion" folds in the Conclusion (as in Table 3).
    inv = pl.read_parquet(INVENTORY_PARQUET, columns=["pmid", "section", "word_count"])
    fields = df.select(pl.col("pmid").cast(pl.Utf8), "journal_field")
    inv = inv.with_columns(pl.col("pmid").cast(pl.Utf8))
    swc_rows: list[dict] = []
    for name, secs in LENGTH_UNITS.items():
        wc = (inv.filter(pl.col("section").is_in(secs))
              .group_by("pmid").agg(pl.col("word_count").sum())
              .join(fields, on="pmid", how="inner"))
        for field in ("all", "medical_education", "general_medicine"):
            sub = wc if field == "all" else wc.filter(pl.col("journal_field") == field)
            # Section length is computed over articles where the section is present.
            vals = sub.filter(pl.col("word_count") > 0)["word_count"]
            if vals.len() == 0:
                continue
            swc_rows.append({
                "section": name, "field": field, "n": vals.len(),
                "median": float(vals.median()),
                "q1": float(vals.quantile(0.25, "linear")),
                "q3": float(vals.quantile(0.75, "linear")),
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
