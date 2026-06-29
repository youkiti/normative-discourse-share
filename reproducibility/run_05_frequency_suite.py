"""Step 05 -- (OPTIONAL, secondary) corpus-pooled G-squared frequency analysis.

EN: The older "bag of words" frequency comparison (G-squared test + rate ratios)
    that pools all words per field. It is NOT the manuscript's primary analysis
    (Tables 2 & 3 come from the R meta-analysis in step 04). Provided so reviewers
    can reproduce the secondary numbers and the by-journal descriptives.
JA: 旧来の語プール型頻度比較（G²検定）。本文の primary（Table 2/3）ではない（それは
    step 04 の R メタ）。副次数値と雑誌別記述統計の再現用。

Core logic reviewed in:
    src/normative_discourse/stats/frequency_analysis.py
    scripts/rerun_frequency_suite_2026-06-26.sh   (this is the OS-agnostic port)

Inputs  : data/processed/section_inventory.parquet,
          data/processed/articles_corpus_10yr_clean.parquet
Outputs : results/output/<section>/        (per-section, ALL articles)
          results/output/sg_<section>/      (IMRaD-only subgroups)
          results/output/primary_abstract_body/  (IMRaD-only, abstract+body combined)

    python reproducibility/run_05_frequency_suite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import INVENTORY_PARQUET, py_cli  # noqa: E402

PER_SECTION = ["abstract_pubmed", "introduction", "methods", "results", "discussion", "conclusion"]
SUBGROUPS = ["abstract_pubmed", "introduction", "methods", "results", "discussion"]
COMBINED = "abstract_pubmed,introduction,methods,results,discussion,conclusion"


def main() -> int:
    if not INVENTORY_PARQUET.exists():
        print(f"[FAIL] {INVENTORY_PARQUET} not found. Run run_03_detect_deontic.py first.")
        return 1

    # Per-section, all articles.
    for sec in PER_SECTION:
        py_cli("analyze-frequency", "--section", sec, label=f"per-section (all): {sec}")

    # IMRaD-only subgroups.
    for sec in SUBGROUPS:
        py_cli(
            "analyze-frequency",
            "--section",
            sec,
            "--imrad-only",
            "--output-dir",
            f"results/output/sg_{sec}",
            label=f"subgroup (IMRaD-only): sg_{sec}",
        )

    # Combined Discussion + Conclusion subgroup.
    py_cli(
        "analyze-frequency",
        "--section",
        "discussion_conclusion",
        "--sections",
        "discussion,conclusion",
        "--imrad-only",
        "--output-dir",
        "results/output/sg_discussion_conclusion",
        label="subgroup (IMRaD-only): sg_discussion_conclusion",
    )

    # Primary combined (abstract + body), IMRaD-only.
    py_cli(
        "analyze-frequency",
        "--section",
        "primary_abstract_body",
        "--sections",
        COMBINED,
        "--imrad-only",
        "--output-dir",
        "results/output/primary_abstract_body",
        label="primary (IMRaD-only, abstract+body combined)",
    )
    print("\nFrequency suite complete -> results/output/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
