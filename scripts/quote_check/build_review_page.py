"""Build the artifact review page with passages and Sonnet drafts in sample order.

Usage::

    python scripts/quote_check/build_review_page.py --sonnet-dir path/to/sonnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from normative_discourse.config.settings import PROJECT_ROOT

TEMPLATE = Path(__file__).with_name("page_template.html")


def build_review_page(sample_path: Path, sonnet_dir: Path, out: Path) -> None:
    """Embed only the review fields; escape closing tags inside the JSON."""
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    drafts = []
    for path in sorted(sonnet_dir.glob("out_*.json")):
        drafts.extend(json.loads(path.read_text(encoding="utf-8")))
    if [r["id"] for r in sample] != [r["id"] for r in drafts]:
        raise SystemExit("Sonnet output ids must match the complete sample order")
    records = []
    for record, draft in zip(sample, drafts):
        item = {key: record[key] for key in ("id", "before", "marker", "after")}
        item["sonnet"] = {key: draft[key] for key in ("speaker", "reason", "evidence")}
        records.append(item)
    payload = json.dumps(records, ensure_ascii=False, indent=1).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__RECORDS__", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sample.json")
    ap.add_argument("--sonnet-dir", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sonnet")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/speaker_review.html")
    args = ap.parse_args()
    build_review_page(args.sample, args.sonnet_dir, args.out)
    print(f"wrote review page to {args.out}")


if __name__ == "__main__":
    main()
