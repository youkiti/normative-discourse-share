"""Reference step -- screen titles and PubMed abstracts for eligibility.

EN: Reference code only, not run by run_all.py: requires a Gemini API key and
    network access, uses a preview model, and produces nondeterministic decisions.
    The main reproduction path uses frozen screening decisions instead.
JA: タイトル・PubMed 抄録の適格性判定を行う参照用コード。API キーとネット接続が
    必要で、プレビューモデルの判定は非決定的なため run_all.py では実行しない。
    主再現経路では凍結済みの判定を使用する。

Input / 入力: Unfiltered corpus Parquet / 判定前コーパス
              (pmid, title, abstract_pubmed; journal also needed for review)
Outputs / 出力: reproducibility/work/screening_results_rerun.csv
                reproducibility/work/screening_review_sample_rerun.tsv (optional / 任意)

    python reproducibility/ref_screen_abstracts.py --input PATH_TO_PRESCREEN_PARQUET
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import REPO_ROOT, SRC_DIR  # noqa: E402

sys.path.insert(0, str(SRC_DIR))

WORK = REPO_ROOT / "reproducibility" / "work"


def _is_frozen_input(path: Path) -> bool:
    resolved = path.resolve()
    processed = (REPO_ROOT / "data" / "processed").resolve()
    return resolved.parent == processed and (
        resolved.name.lower() == "screening_results.csv"
        or resolved.match("screening_*review*.tsv")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Unfiltered Parquet with pmid, title, abstract_pubmed (and journal for review)",
    )
    parser.add_argument("--output", type=Path, default=WORK / "screening_results_rerun.csv")
    parser.add_argument(
        "--review-output", type=Path, default=WORK / "screening_review_sample_rerun.tsv",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--generate-review", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Screen only the first 5 pending articles using the API; skip review sampling",
    )
    args = parser.parse_args()

    for path in (args.output, args.review_output):
        if _is_frozen_input(path):
            print(f"[FAIL] Refusing to overwrite frozen screening input: {path}")
            return 1
    if not args.input.exists():
        print(f"[FAIL] Input not found: {args.input}")
        return 1
    if args.concurrency < 1:
        print("[FAIL] --concurrency must be at least 1")
        return 1

    from normative_discourse.config.settings import get_settings

    settings = get_settings()
    errors = settings.validate_required("screening")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    from normative_discourse.data_io.abstract_screener import (
        generate_review_sample,
        run_batch_screening,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    results = run_batch_screening(
        args.input,
        args.output,
        settings.gemini_api_key,
        resume=not args.no_resume,
        concurrency=args.concurrency,
        limit=5 if args.dry_run else None,
    )
    ok = results["status"] == "ok"
    included = (ok & (results["decision"] == "include")).sum()
    excluded = (ok & (results["decision"] == "exclude")).sum()
    errors = (results["status"] == "error").sum()
    print(f"Screening: include={included}, exclude={excluded}, error={errors}")
    if args.output.exists():
        print(f"Wrote screening results: {args.output}")
    else:
        print("No articles to screen; no screening output written.")
    if args.generate_review and not args.dry_run and args.output.exists():
        generate_review_sample(args.output, args.review_output, args.input, seed=args.seed)
        print(f"Wrote review sample: {args.review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
