#!/usr/bin/env python
"""Strip trailing reference / back-matter blocks leaked into the `discussion`
section of the IMRaD corpus (PDF-derived contamination).

Strategy (see project memory / discussion):
  - Adobe PDF extraction drops the "References" heading, so a keyword split
    does NOT work (heading present in only ~3/234 flagged cases).
  - Instead, the discussion is split into `\n\n`-delimited paragraphs and the
    TRAILING block of reference-entry / back-matter paragraphs is detected and
    removed. Each reference is one paragraph with a stable signature
    (`Year; vol: page-page`, author-initial runs, etc.).

Default mode is DRY-RUN: it computes diagnostics + a diff report and writes a
new parquet only when --apply is given (never overwrites the input).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------
# Paragraph classifiers
# --------------------------------------------------------------------------

# Year; vol: page-page   e.g. "2012;366:1051-1056"  "2010; 32: 638-645"
REFSIG_VOLPAGE = re.compile(r"(?:19|20)\d\d\s*;\s*\d{1,4}\s*:\s*\d{1,4}\s*[-–]\s*\d{1,4}")
# Year; vol:             e.g. "2015;83:" (entry truncated)
REFSIG_VOLONLY = re.compile(r"(?:19|20)\d\d\s*;\s*\d{1,4}\s*:")
# author-initial run, surname-first: two+ of "Lastname AB, " / "Lastname A. "
AUTHORRUN = re.compile(r"(?:[A-Z][A-Za-z'’\-]+ (?:[A-Z]\.?\s?){1,4},?\s){2,}")
# strong author run (3+ "Surname II," tokens): citation-exclusive, no year needed
AUTHORRUN_STRONG = re.compile(r"(?:[A-Z][A-Za-z'’\-]+ (?:[A-Z]\.?\s?){1,4},\s){3,}")
# author run, initials-first (AIM contribution/ref style): "F. Galbiati, I. Remba-Shapiro,"
AUTHORRUN_INIT = re.compile(r"(?:[A-Z]\.\s?[A-Z][A-Za-z'’\-]+,?\s?){2,}")
YEAR = re.compile(r"\b(?:19|20)\d\d\b")
ETAL = re.compile(r"\bet al\.?")
# continuation fragment of a split reference: leads with "vol: page-page"
CONT_VOLPAGE = re.compile(r"^\s*\d{1,4}\s*:\s*\d{1,4}\s*[-–]\s*\d{1,4}")
# paragraph STARTS with an author token "Surname AB," / "Surname A." (incl.
# book/report cites with institutional authors). Anchored => low false-positive
# on prose, which almost never opens with surname+initials.
START_AUTHOR = re.compile(r"^\s*[A-Z][A-Za-z'’\-]{1,}[,]?\s(?:[A-Z]\.?\s?){1,4}[A-Z]?[,.]")
PMID_TAG = re.compile(r"\[PMID:?\s*\d+\]|\bdoi:\s*10\.|https?://doi\.org")

# Back-matter section headings / boilerplate (paragraph-leading or whole-para)
BACKMATTER = re.compile(
    r"(?i)^\s*(?:"
    r"acknowledge?ments?|"
    r"declarations? of (?:interest|competing)|"
    r"conflicts? of interest|"
    r"competing interests?|"
    r"disclosures?|"
    r"funding(?: source| statement)?|"
    r"role of the funding source|"
    r"contributors?|"
    r"author(?:s'?)? contributions?|"
    r"supplementary|supporting information|"
    r"data sharing|data availability|"
    r"this is an open[\s-]access article|"
    r"creative commons|copyright ©|©\s?\d{4}"
    r")\b"
)
# Author-contribution sentence templates (ICMJE / Ann Intern Med style)
AUTHORCONTRIB = re.compile(
    r"(?i)\b(?:"
    r"final approval of the article|"
    r"analysis and interpretation of the data|"
    r"drafting of the article|"
    r"critical revision (?:for|of)|"
    r"provision of study materials|"
    r"collection and assembly of data|"
    r"administrative,? ?technical,? ?or ?logistic support|"
    r"obtaining of funding|"
    r"conception and design"
    r")\b"
)


def classify(p: str) -> str | None:
    """Return 'backmatter' / 'ref' / None for a single paragraph."""
    s = p.strip()
    if not s:
        return "blank"
    if BACKMATTER.search(s) or AUTHORCONTRIB.search(s):
        return "backmatter"
    if REFSIG_VOLPAGE.search(s) or REFSIG_VOLONLY.search(s):
        return "ref"
    # continuation fragment of a reference split by the PDF extractor
    if CONT_VOLPAGE.search(s):
        return "ref"
    # PMID/DOI tag is citation-exclusive (Ann Intern Med reference style)
    if PMID_TAG.search(s):
        return "ref"
    # strong author run (3+ authors) => reference even if the year/volume was
    # split off into a continuation paragraph by the PDF extractor
    if AUTHORRUN_STRONG.search(s) and len(s.split()) < 70:
        return "ref"
    nw = len(s.split())
    # starts with an author token + has a year => reference (any length)
    if START_AUTHOR.search(s) and YEAR.search(s):
        return "ref"
    # author run (either order) + year, reasonably short = citation entry
    if (AUTHORRUN.search(s) or AUTHORRUN_INIT.search(s)) and YEAR.search(s) and nw < 80:
        return "ref"
    # NB: an "et al." + year rule was deliberately dropped -- it mis-classified
    # genuine prose with in-text citations like "Kern et al. (2009)".
    return None


def find_cut(disc: str, min_density: float = 0.6, min_ref: int = 3):
    """Find the index of the first paragraph of the trailing reference block.

    Reference lists contain un-classifiable continuation lines / book cites
    (label None) interleaved with clear reference paragraphs, so a contiguous
    backward walk breaks too early. Instead we exploit tail monotonicity (H4):
    once references begin, the remainder is reference-dominated.

      1. find the EARLIEST index i0 such that, among non-blank paragraphs in
         paras[i0:], the fraction classified ref/backmatter >= `min_density`
         and the tail holds >= `min_ref` reference paragraphs;
      2. advance the cut to the first ref/backmatter paragraph at/after i0 so
         the last *kept* paragraph is never a reference and prose is preserved.

    Returns (cut_index, labels) or (None, labels).
    """
    paras = re.split(r"\n\n+", disc)
    labels = [classify(p) for p in paras]
    n = len(paras)
    wlen = [len(p.split()) for p in paras]
    short = [w < 25 for w in wlen]

    def is_refish(j):
        return labels[j] in ("ref", "backmatter") or (labels[j] is None and short[j])

    # Forward scan for the START of the trailing contamination block. The
    # discussion is the LEADING prose run; once the reference/back-matter block
    # begins it continues to the end (later appendix legends / author addresses
    # are also contamination). A local WINDOW density guard prevents triggering
    # on an isolated in-text citation (a lone ref surrounded by prose) and on a
    # mis-classified opening sentence (prose immediately follows -> low window).
    W = 6
    for k in range(n):
        if labels[k] not in ("ref", "backmatter"):
            continue
        win = range(k, min(n, k + W))
        nb = [j for j in win if labels[j] != "blank"]
        if not nb:
            continue
        dens = sum(1 for j in nb if is_refish(j)) / len(nb)
        if dens < min_density:
            continue
        # the suffix from k must still hold enough references overall
        if sum(1 for l in labels[k:] if l == "ref") < min_ref:
            continue
        return k, labels

    # Fallback: a trailing pure back-matter block (author contributions /
    # disclosures / funding) with NO reference list. Cut at the earliest
    # back-matter paragraph whose suffix is back-matter only (allowing short
    # continuation fragments / blanks) -- i.e. no genuine long prose follows.
    for b in range(1, n):
        if labels[b] != "backmatter":
            continue
        if all(labels[j] == "backmatter" or labels[j] == "blank"
               or (labels[j] is None and short[j]) for j in range(b, n)):
            return b, labels
    return None, labels


def apply_cut(disc: str, cut: int):
    paras = re.split(r"\n\n+", disc)
    kept = "\n\n".join(paras[:cut]).rstrip()
    removed = "\n\n".join(paras[cut:]).strip()
    return kept, removed


# --------------------------------------------------------------------------
# Contamination detector (same heuristic used to flag the problem originally)
# --------------------------------------------------------------------------
VOL_PAGES = re.compile(r"\b\d{1,4}[:;]\s?\d{1,4}[-–]\d{1,4}")
REFLIST_RUN = re.compile(r"(?:[A-Z][a-z]+ [A-Z]{1,3}, ){2,}")


def is_flagged(disc: str) -> bool:
    if not disc:
        return False
    return (
        len(VOL_PAGES.findall(disc)) >= 5
        or len(AUTHORCONTRIB.findall(disc)) >= 2
        or len(REFLIST_RUN.findall(disc)) >= 4
    )


def wc(s: str | None) -> int:
    return 0 if not s else len(s.split())


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/articles_corpus_10yr_draft.parquet")
    ap.add_argument("--output", default="data/processed/articles_corpus_10yr_clean.parquet")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--apply", action="store_true", help="write cleaned parquet (default: dry-run only)")
    args = ap.parse_args()

    df = pl.read_parquet(args.input)
    imrad_mask = ~pl.col("is_non_imrad")
    n_total = df.height
    n_imrad = df.filter(imrad_mask).height

    cut_rows = []        # diff records
    fp_rows = []         # false-positive candidates (clean-source cuts)
    removed_words_all = []

    new_disc = []
    new_ctf = []
    new_wc = []
    new_body = []

    cols = ["pmid", "journal", "source_format", "is_non_imrad",
            "discussion", "citation_tables_figures",
            "introduction", "methods", "results", "conclusion", "other_sections",
            "abstract", "word_count"]
    data = df.select(cols)

    for (pmid, journal, sfmt, non_imrad, disc, ctf, intro, methods,
         results, concl, other, abstract, old_wc) in data.iter_rows():

        keep_disc = disc
        removed = None
        if (not non_imrad) and disc:
            cut, labels = find_cut(disc)
            if cut is not None and cut < len(re.split(r"\n\n+", disc)):
                kept, removed = apply_cut(disc, cut)
                keep_disc = kept
                rw = wc(removed)
                removed_words_all.append(rw)
                rec = {
                    "pmid": pmid, "journal": journal, "source_format": sfmt,
                    "flagged_before": is_flagged(disc),
                    "disc_words_before": wc(disc),
                    "disc_words_after": wc(kept),
                    "removed_words": rw,
                    "removed_frac": round(rw / max(1, wc(disc)), 3),
                    "last_kept_tail": (kept[-180:] if kept else ""),
                    "first_removed": removed[:180],
                }
                cut_rows.append(rec)
                if sfmt in ("pmc_xml", "pmc_html"):
                    fp_rows.append(rec)

        # rebuild derived fields
        new_disc.append(keep_disc)
        if removed:
            merged_ctf = ((ctf or "") + "\n\n" + removed).strip()
        else:
            merged_ctf = ctf
        new_ctf.append(merged_ctf)
        # body_full = sections joined exactly as corpus_builder does (intro,
        # methods, results, discussion, conclusion, other_sections; non-None)
        body_parts = [x for x in (intro, methods, results, keep_disc, concl, other) if x]
        new_body.append("\n\n".join(body_parts) if body_parts else None)
        # word_count = abstract + body sections (corpus_builder definition).
        # Only `discussion` changed, so subtract exactly the removed words; this
        # leaves every uncut row's word_count byte-identical.
        new_wc.append(old_wc - wc(removed) if removed else old_wc)

    # ---- recall check: re-flag after cleaning ----
    flagged_before = sum(1 for d in data.select("discussion").to_series()
                         if (not None) and is_flagged(d) ) if False else None
    fb = sum(1 for r, d in zip(data.select("is_non_imrad").to_series(),
                               data.select("discussion").to_series())
             if (not r) and is_flagged(d))
    fa = sum(1 for r, d in zip(data.select("is_non_imrad").to_series(), new_disc)
             if (not r) and is_flagged(d))

    # ---- summary ----
    import statistics as st
    print("=" * 64)
    print(f"input: {args.input}")
    print(f"total rows: {n_total} | IMRaD: {n_imrad}")
    print(f"discussions cut: {len(cut_rows)}")
    if removed_words_all:
        print(f"removed words: total={sum(removed_words_all)} "
              f"median={int(st.median(removed_words_all))} "
              f"mean={int(st.mean(removed_words_all))} "
              f"max={max(removed_words_all)}")
    print()
    print(f"[Recall]  flagged-contaminated IMRaD discussions: before={fb}  after={fa}")
    print(f"[Specificity] cuts on clean pmc_xml/html sources : {len(fp_rows)} "
          f"(should be ~0)")
    # source-format breakdown of cuts
    from collections import Counter
    print("[Cuts by source_format]:", dict(Counter(r["source_format"] for r in cut_rows)))
    print("[Cuts by journal]:", dict(Counter(r["journal"] for r in cut_rows).most_common()))
    # over-cut watch
    overcut = [r for r in cut_rows if r["removed_frac"] > 0.7]
    tiny = [r for r in cut_rows if r["removed_words"] < 30]
    print(f"[Watch] removed_frac>0.70: {len(overcut)} | removed_words<30: {len(tiny)}")

    # ---- write reports ----
    rdir = Path(args.report_dir)
    rdir.mkdir(exist_ok=True)
    report = rdir / "discussion_reference_strip_dryrun.json"
    report.write_text(json.dumps({
        "input": args.input,
        "n_total": n_total, "n_imrad": n_imrad,
        "n_cut": len(cut_rows),
        "removed_words_total": sum(removed_words_all) if removed_words_all else 0,
        "recall_flagged_before": fb, "recall_flagged_after": fa,
        "specificity_clean_source_cuts": len(fp_rows),
        "cuts": cut_rows,
        "clean_source_cut_examples": fp_rows[:50],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {report}")

    if args.apply:
        out = df.with_columns([
            pl.Series("discussion", new_disc),
            pl.Series("citation_tables_figures", new_ctf),
            pl.Series("body_full", new_body),
            pl.Series("word_count", new_wc),
        ])
        out.write_parquet(args.output)
        print(f"APPLIED -> {args.output}")
    else:
        print("\n(dry-run; no parquet written. pass --apply to write cleaned corpus)")


if __name__ == "__main__":
    main()
