"""Step 01 -- build the analysis corpus from cleaned_text JSON.

EN: Assembles per-article section text (data/interim/cleaned_text/*.json) +
    metadata + cached PubMed abstracts + screening filter + manual section
    overrides into a single Parquet (the "draft" corpus).
JA: cleaned_text JSON とメタデータ・抄録キャッシュ・スクリーニング・節 override を
    1 つの Parquet（draft コーパス）に統合する。

Core logic reviewed in:
    src/normative_discourse/data_io/corpus_builder.py
    config/section_overrides.toml   (per-article section reassignments)

Inputs  : data/interim/cleaned_text/*.json, data/interim/pubmed_abstracts.json,
          data/raw/pubmed/articles_2016_2025_included.csv,
          data/processed/screening_results.csv (+ 2 review TSVs),
          data/processed/med_teach_article_types.csv, config/section_overrides.toml
Output  : data/processed/articles_corpus_10yr_draft.parquet

Footguns guarded here:
  * `--articles-csv` MUST point at the 10-year manifest; the CLI default is a
    different 1,314-row corpus.
  * The abstract cache MUST exist; otherwise build-corpus calls PubMed over the
    network. We re-check and abort before launching.

    python reproducibility/run_01_build_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import DRAFT_PARQUET, check_inputs, print_check, py_cli  # noqa: E402

# Repo-relative paths (run() executes with cwd=REPO_ROOT, so these resolve there).
ARTICLES_CSV_REL = "data/raw/pubmed/articles_2016_2025_included.csv"
DRAFT_PARQUET_REL = "data/processed/articles_corpus_10yr_draft.parquet"


def main() -> int:
    check = check_inputs()
    print_check(check)
    if not check.ok:
        print("\nInputs incomplete -- fix the above before building. Aborting.")
        return 1

    py_cli(
        "build-corpus",
        "--articles-csv",
        ARTICLES_CSV_REL,
        "--output",
        DRAFT_PARQUET_REL,
        label="build-corpus (10-year manifest, abstract cache, screening, overrides)",
    )
    print(f"\nWrote draft corpus: {DRAFT_PARQUET}")
    print("Next: python reproducibility/run_02_strip_references.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
