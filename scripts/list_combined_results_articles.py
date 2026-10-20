"""List analysed articles whose Results may include Discussion text.

Usage::

    python scripts/list_combined_results_articles.py
    python scripts/list_combined_results_articles.py --cleaned-dir path/to/json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

from normative_discourse.config.settings import PROJECT_ROOT

COMBINED_HEADING = re.compile(
    r"(result|finding)s?\s*(and|&)\s*discussion|discussion\s*(and|&)\s*(result|finding)",
    re.IGNORECASE,
)
OUTPUT_COLUMNS = ["pmid", "journal", "journal_field", "reason", "heading"]


def list_combined_articles(
    cleaned_dir: Path, corpus_path: Path, inventory_path: Path
) -> tuple[pl.DataFrame, int]:
    """Return exclusions and the count of missing or unreadable JSON files."""
    corpus = (
        pl.read_parquet(corpus_path)
        .select("pmid", "journal", "journal_field", "is_non_imrad")
        .with_columns(pl.col("pmid").cast(pl.String))
        .filter(~pl.col("is_non_imrad").fill_null(False))
        .unique(subset=["pmid"])
    )
    inventory = (
        pl.read_parquet(inventory_path)
        .with_columns(pl.col("pmid").cast(pl.String))
        .group_by("pmid")
        .agg(
            pl.col("word_count").filter(pl.col("section") == "results").sum().alias("results"),
            pl.col("word_count")
            .filter(pl.col("section").is_in(["discussion", "conclusion"]))
            .sum()
            .alias("discussion_conclusion"),
        )
    )
    no_discussion = set(
        inventory.filter(
            (pl.col("results") > 0) & (pl.col("discussion_conclusion") == 0)
        )["pmid"].to_list()
    )
    rows = []
    unreadable = 0
    for article in corpus.iter_rows(named=True):
        pmid = article["pmid"]
        heading = ""
        try:
            payload = json.loads((cleaned_dir / f"{pmid}.json").read_text(encoding="utf-8"))
            for section in payload.get("sections", []):
                title = section.get("section_title") or ""
                if COMBINED_HEADING.search(title):
                    heading = title
                    break
        except (OSError, ValueError, TypeError, AttributeError):
            unreadable += 1
        missing_discussion = pmid in no_discussion
        if not heading and not missing_discussion:
            continue
        reason = "both" if heading and missing_discussion else (
            "combined_heading" if heading else "no_discussion_section"
        )
        rows.append({
            "pmid": pmid,
            "journal": article["journal"],
            "journal_field": article["journal_field"],
            "reason": reason,
            "heading": heading,
        })
    schema = {column: pl.String for column in OUTPUT_COLUMNS}
    return pl.DataFrame(rows, schema=schema).sort("pmid"), unreadable


def print_summary(articles: pl.DataFrame, unreadable: int) -> None:
    """Print counts even when no articles match."""
    for reason in ("combined_heading", "no_discussion_section", "both"):
        print(f"reason {reason}: {articles.filter(pl.col('reason') == reason).height}")
    for field in ("general_medicine", "medical_education"):
        print(f"journal_field {field}: {articles.filter(pl.col('journal_field') == field).height}")
    print(f"Missing or unreadable JSON files: {unreadable}")
    print(f"Total: {articles.height}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned-dir", type=Path, default=PROJECT_ROOT / "data/interim/cleaned_text")
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/processed/articles_corpus_10yr_clean.parquet")
    ap.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data/processed/section_inventory.parquet")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/output_sensitivity/combined_results_articles.csv")
    args = ap.parse_args()

    articles, unreadable = list_combined_articles(args.cleaned_dir, args.corpus, args.inventory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    articles.write_csv(args.out)
    print_summary(articles, unreadable)


if __name__ == "__main__":
    main()
