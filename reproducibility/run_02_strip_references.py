"""Step 02 -- strip leaked reference / back-matter from the Discussion section.

EN: PDF-derived Discussion text often has trailing reference entries appended
    (Adobe drops the "References" heading). This removes the trailing reference
    block paragraph-by-paragraph and writes the FROZEN clean corpus. It never
    overwrites its input.
JA: PDF 由来の Discussion 末尾に紛れ込む参考文献ブロックを段落単位で除去し、frozen な
    clean コーパスを書き出す。入力は上書きしない。

Core logic reviewed in:
    scripts/strip_discussion_references.py

Input   : data/processed/articles_corpus_10yr_draft.parquet
Output  : data/processed/articles_corpus_10yr_clean.parquet  (the frozen corpus)
          + a diff report under reports/

    python reproducibility/run_02_strip_references.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import CLEAN_PARQUET, DRAFT_PARQUET, run  # noqa: E402


def main() -> int:
    if not DRAFT_PARQUET.exists():
        print(f"[FAIL] draft corpus not found: {DRAFT_PARQUET}")
        print("Run run_01_build_corpus.py first.")
        return 1

    run(
        [
            sys.executable,
            "scripts/strip_discussion_references.py",
            "--input",
            "data/processed/articles_corpus_10yr_draft.parquet",
            "--output",
            "data/processed/articles_corpus_10yr_clean.parquet",
            "--apply",
        ],
        label="strip discussion references (writes frozen clean corpus)",
    )
    print(f"\nWrote frozen corpus: {CLEAN_PARQUET}")
    print("Next: python reproducibility/run_03_detect_deontic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
