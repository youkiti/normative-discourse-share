"""Shared helpers for the reproducibility wrappers.

EN: Central place for (a) locating the repository root, (b) running the existing
    package / scripts as subprocesses with a correct, reproducible environment,
    and (c) checking that the Google-Drive-shared input data is in place.

JA: 再現用ラッパの共通ヘルパ。(a) リポジトリルートの特定、(b) 既存パッケージ /
    スクリプトを再現可能な環境変数つきで subprocess 実行、(c) Google Drive で共有された
    入力データが所定の場所に在るかの検査、を一元化する。

Why this file exists (design notes from the plan review):
  * PYTHONPATH=<repo>/src  -- the wrappers import the *existing* package without
    `pip install -e .` (which would drag in selenium / pdfservices / google-genai).
  * PYTHONUTF8=1 / PYTHONIOENCODING=utf-8 -- the Rich console output otherwise
    crashes on Windows for some tables.
  * Network must NOT be touched: build-corpus fetches PubMed abstracts unless the
    cache file is present, so we hard-fail when the cache is missing rather than
    silently going online.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# reproducibility/_env.py -> repository root is the parent directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# --- Standard paths the existing CLI/scripts resolve by default ---------------
ARTICLES_CSV = REPO_ROOT / "data" / "raw" / "pubmed" / "articles_2016_2025_included.csv"
CLEANED_TEXT_DIR = REPO_ROOT / "data" / "interim" / "cleaned_text"
ABSTRACT_CACHE = REPO_ROOT / "data" / "interim" / "pubmed_abstracts.json"
SCREENING_CSV = REPO_ROOT / "data" / "processed" / "screening_results.csv"
MED_TEACH_CSV = REPO_ROOT / "data" / "processed" / "med_teach_article_types.csv"
SECTION_OVERRIDES = REPO_ROOT / "config" / "section_overrides.toml"

DRAFT_PARQUET = REPO_ROOT / "data" / "processed" / "articles_corpus_10yr_draft.parquet"
CLEAN_PARQUET = REPO_ROOT / "data" / "processed" / "articles_corpus_10yr_clean.parquet"
INVENTORY_PARQUET = REPO_ROOT / "data" / "processed" / "section_inventory.parquet"
MATCHES_PARQUET = REPO_ROOT / "data" / "processed" / "matches.parquet"
META_OUT_DIR = REPO_ROOT / "results" / "output_meta"

# build-corpus globs screening_*review*.tsv and reads ALL matches; only these two
# are expected. Any extra file would silently change the screening filter.
EXPECTED_REVIEW_TSVS = {
    "screening_review_sample.tsv",
    "screening_excluded_review.tsv",
}

# The six IMRaD-ish sections detection/meta operate on (in the canonical order).
SECTIONS = [
    "abstract_pubmed",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
]


def subprocess_env() -> dict[str, str]:
    """Environment for child processes: import path + UTF-8 console."""
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)
    )
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(args: list[str], *, label: str | None = None) -> None:
    """Run a command from the repo root, streaming output; raise on failure."""
    if label:
        print(f"\n=== {label} ===", flush=True)
    print(f"$ {' '.join(args)}", flush=True)
    result = subprocess.run(args, cwd=str(REPO_ROOT), env=subprocess_env())
    if result.returncode != 0:
        raise SystemExit(f"[FAIL] command exited with {result.returncode}: {' '.join(args)}")


def py_cli(*cli_args: str, label: str | None = None) -> None:
    """Run `python -m normative_discourse.cli.main <cli_args>`."""
    run([sys.executable, "-m", "normative_discourse.cli.main", *cli_args], label=label)


@dataclass
class InputCheck:
    ok: bool
    problems: list[str]
    notes: list[str]


def check_inputs(*, require_clean_parquet: bool = False) -> InputCheck:
    """Verify the Drive-shared inputs are placed at the expected paths.

    JA: Google Drive から配置すべき入力が揃っているか検査する。キャッシュ欠落や
        余分な review tsv は致命的エラー（fail）として扱う。
    """
    problems: list[str] = []
    notes: list[str] = []

    # cleaned_text JSON corpus (reproduction starting point).
    if not CLEANED_TEXT_DIR.is_dir():
        problems.append(f"missing dir: {CLEANED_TEXT_DIR}  (place cleaned_text/*.json here)")
    else:
        n = len(list(CLEANED_TEXT_DIR.glob("*.json")))
        if n == 0:
            problems.append(f"no *.json in {CLEANED_TEXT_DIR}")
        else:
            notes.append(f"cleaned_text JSON files: {n}")

    # Abstract cache: REQUIRED. Without it build-corpus goes to the network and
    # ~450 introductions are lost.
    if not ABSTRACT_CACHE.exists():
        problems.append(
            f"missing: {ABSTRACT_CACHE}  (REQUIRED -- without it build-corpus calls "
            f"PubMed over the network and drops ~450 introductions)"
        )
    else:
        notes.append(f"abstract cache present: {ABSTRACT_CACHE.name}")

    # Article manifest.
    if not ARTICLES_CSV.exists():
        problems.append(f"missing: {ARTICLES_CSV}")

    # Screening inputs.
    if not SCREENING_CSV.exists():
        problems.append(f"missing: {SCREENING_CSV}")
    review_tsvs = sorted(
        p.name for p in SCREENING_CSV.parent.glob("screening_*review*.tsv")
    )
    extra = set(review_tsvs) - EXPECTED_REVIEW_TSVS
    missing_review = EXPECTED_REVIEW_TSVS - set(review_tsvs)
    if missing_review:
        problems.append(f"missing screening review TSV(s): {sorted(missing_review)}")
    if extra:
        problems.append(
            f"unexpected screening_*review*.tsv (build-corpus reads ALL of them): {sorted(extra)}"
        )

    # Ships with the repo, but warn if absent.
    for p in (MED_TEACH_CSV, SECTION_OVERRIDES):
        if not p.exists():
            problems.append(f"missing (should ship in repo): {p}")

    if require_clean_parquet and not CLEAN_PARQUET.exists():
        problems.append(f"missing: {CLEAN_PARQUET}  (run run_02_strip_references.py first)")

    return InputCheck(ok=not problems, problems=problems, notes=notes)


def print_check(check: InputCheck) -> None:
    for note in check.notes:
        print(f"  [ok] {note}")
    for prob in check.problems:
        print(f"  [!!] {prob}")
    print("  => INPUTS OK" if check.ok else "  => INPUTS INCOMPLETE")
