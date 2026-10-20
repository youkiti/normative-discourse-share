"""Draw a frozen random sample of deontic markers in Results sections for the quote-vs-author check.

Population: deontic-marker matches in the Results section of analysed articles (IMRaD corpus),
excluding the articles whose Results also carry the Discussion
(results/output_sensitivity/combined_results_articles.csv). Simple random sample without
replacement of N markers per journal field, seed fixed. The sample is written once; a second run
refuses to overwrite it.

Usage::

    python scripts/quote_check/sample_results_markers.py --n-per-field 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl

from normative_discourse.config.settings import PROJECT_ROOT
from normative_discourse.preprocessing.text_normalizer import normalize_for_detection

SEED = 20261018
WINDOW = 600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/processed/articles_corpus_10yr_clean.parquet")
    ap.add_argument("--matches", type=Path, default=PROJECT_ROOT / "data/processed/matches.parquet")
    ap.add_argument("--combined", type=Path, default=PROJECT_ROOT / "results/output_sensitivity/combined_results_articles.csv")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sample.json")
    ap.add_argument("--n-per-field", type=int, default=100)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the sample is frozen and will not be redrawn")

    corpus = pl.read_parquet(args.corpus,
                             columns=["pmid", "journal", "journal_field", "is_non_imrad", "results"])
    corpus = corpus.filter(~pl.col("is_non_imrad").fill_null(False))
    combined = set(pl.read_csv(args.combined,
                               schema_overrides={"pmid": pl.Utf8})["pmid"].to_list())
    matches = pl.read_parquet(args.matches).filter(pl.col("section") == "results")
    pop = (matches.join(corpus.select("pmid", "journal", "journal_field"), on="pmid", how="inner")
           .filter(~pl.col("pmid").is_in(list(combined)))
           .sort(["pmid", "start_char"]))
    texts = dict(corpus.select("pmid", "results").iter_rows())

    rng = random.Random(SEED)
    sample = []
    for field in ["medical_education", "general_medicine"]:
        rows = pop.filter(pl.col("journal_field") == field).to_dicts()
        print(f"{field}: population {len(rows)} markers in {len(set(r['pmid'] for r in rows))} articles")
        for r in rng.sample(rows, args.n_per_field):
            text = normalize_for_detection(texts[r["pmid"]] or "")
            s, e = r["start_char"], r["end_char"]
            if not (0 <= s < e <= len(text)):
                raise SystemExit(f"offset out of range for {r['pmid']} {s}-{e}")
            a = max(0, s - WINDOW)
            b = min(len(text), e + WINDOW)
            sample.append({
                "id": f"q{len(sample):03d}",
                "pmid": r["pmid"],
                "journal": r["journal"],
                "journal_field": field,
                "expression": r["expression"],
                "expression_group": r["expression_group"],
                "start_char": s,
                "end_char": e,
                "before": text[a:s],
                "marker": text[s:e],
                "after": text[e:b],
            })
    rng.shuffle(sample)
    for i, rec in enumerate(sample):
        rec["id"] = f"q{i:03d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(sample)} records to {args.out}")


if __name__ == "__main__":
    main()
