"""Export public identifiers, marker offsets and speaker labels from private ratings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tally_speaker import CATS_V1, FIELDS, LABEL_COLUMNS, SONNET_V1_DIR, canonical_sha256, load_human, load_sonnet
from normative_discourse.config.settings import PROJECT_ROOT


def export_labels(sample_path: Path, sonnet_dir: Path, sonnet_v1_dir: Path,
                  human_dir: Path, out: Path, overwrite: bool = False) -> None:
    """Validate private ratings and write only the fixed public allowlist."""
    meta_path = out.with_suffix(".meta.json")
    for path in (out, meta_path):
        if path.exists() and not overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    raw = sample_path.read_bytes()
    sample = json.loads(raw)
    if any(r["journal_field"] not in FIELDS for r in sample):
        raise SystemExit("unknown journal field in sample")
    sonnet, bad = load_sonnet(sample, sonnet_dir)
    v1, bad_v1 = load_sonnet(sample, sonnet_v1_dir, CATS_V1) if sonnet_v1_dir.exists() else ({}, [])
    auto_ids = set()
    human = load_human(human_dir, set(sonnet), auto_ids)
    rows = []
    for record in sample:
        record_id = record["id"]
        row = {column: record[column] for column in LABEL_COLUMNS[:8]}
        row.update(sonnet_v1_speaker=v1.get(record_id, ""), sonnet_speaker=sonnet[record_id],
                   human_speaker=human.get(record_id, ""),
                   human_auto="true" if record_id in auto_ids else "false")
        rows.append(row)
    meta = {
        "sample_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_sha256_lf": canonical_sha256(raw),
        "seed": 20261018,
        "n_per_field": 100,
        "window": 600,
        "sonnet_evidence_unverifiable_ids": bad,
        "sonnet_v1_evidence_unverifiable_ids": bad_v1,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    private = PROJECT_ROOT / "data/processed/quote_check"
    ap.add_argument("--sample", type=Path, default=private / "sample.json")
    ap.add_argument("--sonnet-dir", type=Path, default=private / "sonnet")
    ap.add_argument("--sonnet-v1-dir", type=Path, default=SONNET_V1_DIR)
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/quote_check/speaker_labels.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    export_labels(args.sample, args.sonnet_dir, args.sonnet_v1_dir, args.human, args.out, args.overwrite)


if __name__ == "__main__":
    main()
