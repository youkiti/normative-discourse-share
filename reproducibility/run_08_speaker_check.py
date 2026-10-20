"""Step 08: tally public speaker labels and optionally verify the frozen sample.

EN: Produces the label-only report; with local corpus inputs, checks a redrawn sample.
JA: 公開ラベルから集計し、ローカルのコーパスがあれば再抽出標本の属性とハッシュを照合する。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _env import (  # noqa: E402
    CLEAN_PARQUET, COMBINED_CSV, MATCHES_PARQUET, REPO_ROOT, SPEAKER_LABELS, run,
)

sys.path.insert(0, str(REPO_ROOT / "scripts/quote_check"))

from tally_speaker import canonical_sha256  # noqa: E402

IDENTITY_COLUMNS = [
    "pmid", "journal", "journal_field", "expression", "expression_group", "start_char", "end_char",
]


def check_redraw(sample_path: Path, labels_path: Path) -> bool:
    """Check all sampled identities and the LF-normalized fingerprint."""
    raw = sample_path.read_bytes()
    sample = json.loads(raw)
    with labels_path.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    meta = json.loads(labels_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if "sample_sha256_lf" not in meta:
        print("[FAIL] labels metadata missing sample_sha256_lf; re-export labels")
        return False
    by_id = {r["id"]: r for r in labels}
    ids = [r["id"] for r in sample]
    identities_match = (len(ids) == len(set(ids)) == len(labels) == len(by_id)
                        and set(ids) == set(by_id))
    if identities_match:
        identities_match = all(
            str(record[col]) == by_id[record["id"]][col]
            for record in sample for col in IDENTITY_COLUMNS
        )
    hash_matches = canonical_sha256(raw) == meta["sample_sha256_lf"]
    print(f"[{'PASS' if identities_match else 'FAIL'}] redrawn sample identifiers and offsets")
    print(f"[{'PASS' if hash_matches else 'FAIL'}] redrawn sample SHA-256 (LF-normalized)")
    return identities_match and hash_matches


def main() -> int:
    run([
        sys.executable, "scripts/quote_check/tally_speaker.py", "--labels", str(SPEAKER_LABELS),
        "--report", str(REPO_ROOT / "results/quote_check/summary.md"),
    ], label="speaker check from public labels")
    if not all(p.is_file() for p in (CLEAN_PARQUET, MATCHES_PARQUET, COMBINED_CSV)):
        print("Redraw check skipped: clean corpus, matches.parquet or combined CSV is missing.")
        return 0
    sample_path = HERE / "work/quote_check/sample.json"
    if sample_path.exists():
        sample_path.unlink()
    run([
        sys.executable, "scripts/quote_check/sample_results_markers.py",
        "--corpus", str(CLEAN_PARQUET), "--matches", str(MATCHES_PARQUET),
        "--combined", str(COMBINED_CSV), "--out", str(sample_path),
    ], label="redraw frozen speaker sample")
    return 0 if check_redraw(sample_path, SPEAKER_LABELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
