"""Step 04 -- random-effects meta-analysis (PRIMARY analysis: Table 2 & Table 3).

EN: Fits a Poisson-lognormal random-effects model with the article as the unit of
    analysis (each article = one "study"), field as the moderator. Produces the
    incidence-rate ratio (ME vs GM) per modal group with a 95% CI, for the full
    article (primary, Table 2) and within each IMRaD section (subgroup, Table 3).
JA: 論文を解析単位（各論文 = 1 study）とした Poisson-lognormal 変量効果モデル。全文
    （primary = Table 2）と各 IMRaD 節（subgroup = Table 3）について ME vs GM の率比と
    95%CI を算出する。

Core logic reviewed in:
    scripts/meta_analysis_deontic.R   (pure base R; non-IMRaD excluded at line 117)

Requires: R >= 4.5 with packages `arrow` and `data.table`.
Inputs  : data/processed/section_inventory.parquet,
          data/processed/articles_corpus_10yr_clean.parquet
Outputs : results/output_meta/<unit>/meta_results.csv  for units:
            primary_abstract_body      (Table 2)
            sg_abstract_pubmed         (Table 3: Abstract)
            sg_introduction            (Table 3: Introduction)
            sg_methods                 (Table 3: Methods)
            sg_results                 (Table 3: Results)
            sg_discussion_conclusion   (Table 3: "Discussion" column = discussion+conclusion)
          results/output_meta/session_info.txt   (R sessionInfo for the reviewer)

    python reproducibility/run_04_meta_analysis.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import CLEAN_PARQUET, INVENTORY_PARQUET, META_OUT_DIR, run  # noqa: E402


def main() -> int:
    rscript = shutil.which("Rscript")
    if not rscript:
        print("[FAIL] Rscript not found on PATH. Install R >= 4.5 and re-run.")
        return 1
    if not INVENTORY_PARQUET.exists() or not CLEAN_PARQUET.exists():
        print("[FAIL] need section_inventory.parquet and the clean corpus.")
        print("Run run_03_detect_deontic.py first.")
        return 1

    run(
        [
            rscript,
            "scripts/meta_analysis_deontic.R",
            "data/processed/section_inventory.parquet",
            "data/processed/articles_corpus_10yr_clean.parquet",
            "results/output_meta",
        ],
        label="meta-analysis (Poisson-lognormal RE; Table 2 + Table 3)",
    )

    # Record the R environment for the reviewer.
    META_OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            rscript,
            "-e",
            'writeLines(capture.output(sessionInfo()), '
            '"results/output_meta/session_info.txt")',
        ],
        label="record R sessionInfo",
    )
    print(f"\nWrote meta results under: {META_OUT_DIR}")
    print("Next: python reproducibility/run_06_table1.py  (and run_05 for the optional G-test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
