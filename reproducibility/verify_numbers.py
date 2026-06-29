"""Verify reproduced outputs against the manuscript values (Tables 1-3).

EN: Reads the frozen corpus and the meta-analysis CSVs DIRECTLY (never the Rich
    console output) and checks them against the numbers reported in
    manuscripts/methods_results.md. Exits non-zero on any mismatch.
JA: frozen コーパスとメタ解析 CSV を直接読み、原稿 methods_results.md の値と照合する。
    不一致があれば異常終了する。

    python reproducibility/verify_numbers.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from _env import CLEAN_PARQUET, META_OUT_DIR  # noqa: E402

REL_TOL = 0.01  # 1% relative tolerance for rate ratios (manuscript rounds to 2 dp)

results: list[tuple[bool, str]] = []


def check_int(label: str, got: int | None, expected: int) -> None:
    ok = got == expected
    results.append((ok, f"{label}: got {got}, expected {expected}"))


def check_rr(label: str, got: float | None, expected: float) -> None:
    ok = got is not None and abs(got - expected) <= REL_TOL * expected
    shown = f"{got:.3f}" if got is not None else "None"
    results.append((ok, f"{label}: got {shown}, expected ~{expected}"))


def verify_table1() -> None:
    if not CLEAN_PARQUET.exists():
        results.append((False, f"Table 1: missing {CLEAN_PARQUET}"))
        return
    df = pl.read_parquet(CLEAN_PARQUET).filter(~pl.col("is_non_imrad"))
    check_int("Table1 total (~is_non_imrad)", df.height, 10743)
    by_field = {r["journal_field"]: r["len"] for r in df.group_by("journal_field").len().to_dicts()}
    check_int("Table1 medical_education", by_field.get("medical_education"), 3743)
    check_int("Table1 general_medicine", by_field.get("general_medicine"), 7000)

    pyear = df.with_columns(pl.col("pub_year").cast(pl.Int64))
    check_int("Table1 period 2016-2020",
              pyear.filter(pl.col("pub_year") <= 2020).height, 5206)
    check_int("Table1 period 2021-2025",
              pyear.filter(pl.col("pub_year") >= 2021).height, 5537)

    journals = {
        "Acad Med": 1293, "Med Teach": 964, "Med Educ": 612, "JMIR Med Educ": 521,
        "Perspect Med Educ": 353, "N Engl J Med": 2084, "Lancet": 1500, "JAMA": 1452,
        "BMJ": 1061, "Ann Intern Med": 903,
    }
    jc = {r["journal"]: r["len"] for r in df.group_by("journal").len().to_dicts()}
    for j, exp in journals.items():
        check_int(f"Table1 journal {j}", jc.get(j), exp)

    for col, exp in (("introduction", 10743), ("methods", 10743),
                     ("results", 10743), ("discussion", 10731)):
        present = df.filter(pl.col(col).is_not_null() & (pl.col(col).str.strip_chars() != "")).height
        check_int(f"Table1 presence {col}", present, exp)


def _read_meta(unit: str) -> dict[str, dict]:
    path = META_OUT_DIR / unit / "meta_results.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["expression_group"]] = row
    return out


def verify_table2() -> None:
    m = _read_meta("primary_abstract_body")
    if not m:
        results.append((False, "Table 2: missing results/output_meta/primary_abstract_body/meta_results.csv"))
        return
    expected = {"must": 5.12, "should": 2.27, "need_to": 4.04, "have_to": 3.39,
                "required_to": 1.51, "all": 2.74}
    for grp, exp in expected.items():
        rr = float(m[grp]["rate_ratio"]) if grp in m else None
        check_rr(f"Table2 RR {grp}", rr, exp)
    # Total markers (ME 28,553 + GM 20,387 = 48,940) from the `all` row.
    if "all" in m:
        cme = int(round(float(m["all"]["count_me"])))
        cgm = int(round(float(m["all"]["count_gm"])))
        check_int("Table2 markers ME", cme, 28553)
        check_int("Table2 markers GM", cgm, 20387)
        check_int("Table2 markers total", cme + cgm, 48940)


def verify_table3() -> None:
    # Column -> meta unit. NOTE: the "Discussion" column = discussion + conclusion.
    cols = {
        "Abstract": ("sg_abstract_pubmed", 4.19),
        "Introduction": ("sg_introduction", 2.79),
        "Methods": ("sg_methods", 1.23),
        "Results": ("sg_results", 16.97),
        "Discussion(=disc+concl)": ("sg_discussion_conclusion", 2.05),
    }
    for label, (unit, exp) in cols.items():
        m = _read_meta(unit)
        rr = float(m["all"]["rate_ratio"]) if "all" in m else None
        check_rr(f"Table3 all-groups {label}", rr, exp)


def main() -> int:
    verify_table1()
    verify_table2()
    verify_table3()

    n_pass = sum(1 for ok, _ in results if ok)
    print(f"\n{'=' * 60}")
    for ok, msg in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"{'=' * 60}")
    print(f"  {n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
