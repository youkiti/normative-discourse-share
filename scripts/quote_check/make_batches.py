"""Write ordered batches and individual Sonnet rater prompts from a frozen sample.

Usage::

    python scripts/quote_check/make_batches.py --sample path/to/sample.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from normative_discourse.config.settings import PROJECT_ROOT

BATCH_SIZE = 20
TEMPLATE = Path(__file__).with_name("prompt_template.md")


def make_batches(sample_path: Path, out_dir: Path) -> int:
    """Write twenty records per batch, allowing a shorter final batch."""
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    template = TEMPLATE.read_text(encoding="utf-8")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for start in range(0, len(sample), BATCH_SIZE):
        number = f"{count:02d}"
        batch = [
            {key: record[key] for key in ("id", "before", "marker", "after")}
            for record in sample[start:start + BATCH_SIZE]
        ]
        (out_dir / f"batch_{number}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        prompt = template.replace("NN", number).replace("{DIR}", out_dir.as_posix())
        if len(batch) != BATCH_SIZE:
            prompt = prompt.replace("20 records", f"{len(batch)} records")
            prompt = prompt.replace("20 objects", f"{len(batch)} objects")
        (out_dir / f"prompt_{number}.md").write_text(prompt, encoding="utf-8")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sample.json")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sonnet")
    args = ap.parse_args()
    count = make_batches(args.sample, args.out_dir)
    print(f"wrote {count} batches and prompts to {args.out_dir}")


if __name__ == "__main__":
    main()
