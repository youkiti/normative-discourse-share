"""Step 03 -- detect deontic markers in each section.

EN: Runs the curated regular-expression detector (5 modal groups: must, should,
    need_to, have_to, required_to; affirmative + negated) over the six sections
    of every article in the frozen corpus.
JA: 厳選した正規表現検出器（must/should/need_to/have_to/required_to の 5 群、
    肯定+否定）を frozen コーパス全論文の 6 節に適用する。

Core logic reviewed in:
    src/normative_discourse/detection/deontic_detector.py

Input   : data/processed/articles_corpus_10yr_clean.parquet
Outputs : data/processed/section_inventory.parquet  (per-article-per-section tallies)
          data/processed/matches.parquet            (every individual match + context)

IMPORTANT (why the --force guard):
  detect-deontic RESUMES from an existing section_inventory.parquet (it skips
  (pmid, section) pairs already present) and APPENDS to existing output files.
  Running it on top of stale output (e.g. a reference parquet) would either reuse
  old results or double the rows. So we refuse to run if outputs already exist,
  unless --force is given (which deletes them first for a clean regeneration).

    python reproducibility/run_03_detect_deontic.py            # clean run
    python reproducibility/run_03_detect_deontic.py --force    # delete + regenerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import CLEAN_PARQUET, INVENTORY_PARQUET, MATCHES_PARQUET, SECTIONS, py_cli  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        action="store_true",
        help="delete existing section_inventory.parquet / matches.parquet before running",
    )
    args = ap.parse_args()

    if not CLEAN_PARQUET.exists():
        print(f"[FAIL] frozen corpus not found: {CLEAN_PARQUET}")
        print("Run run_02_strip_references.py first.")
        return 1

    existing = [p for p in (INVENTORY_PARQUET, MATCHES_PARQUET) if p.exists()]
    if existing:
        if not args.force:
            print("[FAIL] detection outputs already exist:")
            for p in existing:
                print(f"        {p}")
            print(
                "  detect-deontic resumes/appends, so reusing these would corrupt the\n"
                "  result. Re-run with --force to delete and regenerate, or move them aside."
            )
            return 1
        for p in existing:
            print(f"  removing {p}")
            p.unlink()

    py_cli(
        "detect-deontic",
        "--sections",
        ",".join(SECTIONS),
        label="detect-deontic (5 modal groups x 6 sections)",
    )
    print(f"\nWrote: {INVENTORY_PARQUET}")
    print(f"Wrote: {MATCHES_PARQUET}")
    print("Next: python reproducibility/run_04_meta_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
