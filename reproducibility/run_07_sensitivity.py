"""Step 07: exclusion of combined sections and leave-one-journal-out analyses.

EN: Lists combined Results articles, then fits sensitivity models in R.
JA: Results と Discussion が結合した論文を列挙し、除外解析と雑誌別除外解析を R で実行する。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _env import (  # noqa: E402
    CLEANED_TEXT_DIR, CLEAN_PARQUET, COMBINED_CSV, INVENTORY_PARQUET, SENS_OUT_DIR, run,
)


def main() -> int:
    rscript = shutil.which("Rscript")
    if not rscript:
        print("[FAIL] Rscript not found on PATH. Install R >= 4.5 and re-run.")
        return 1
    missing = [p for p in (CLEAN_PARQUET, INVENTORY_PARQUET) if not p.is_file()]
    if not CLEANED_TEXT_DIR.is_dir() or not any(CLEANED_TEXT_DIR.glob("*.json")):
        missing.append(CLEANED_TEXT_DIR)
    if missing:
        for path in missing:
            print(f"[FAIL] missing sensitivity input: {path}")
        print("Place the cleaned text and run steps 02 and 03 first.")
        return 1
    run([
        sys.executable, "scripts/list_combined_results_articles.py",
        "--cleaned-dir", str(CLEANED_TEXT_DIR), "--corpus", str(CLEAN_PARQUET),
        "--inventory", str(INVENTORY_PARQUET), "--out", str(COMBINED_CSV),
    ], label="list combined Results articles")
    run([
        rscript, "scripts/sensitivity_deontic.R", str(INVENTORY_PARQUET),
        str(CLEAN_PARQUET), str(COMBINED_CSV), str(SENS_OUT_DIR),
    ], label="sensitivity analyses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
