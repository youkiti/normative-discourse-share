"""Flag sampled markers that sit inside quotation marks or square brackets.

The scan runs over the detector's normalised Results text (the text the offsets point into). The
normaliser removes line breaks, so the paragraph bounds below span the whole Results section. The
text is scanned from its start to the marker, tracking open double quotes (“ or a straight " that
opens), open single quotes (‘ or a straight ' that opens, apostrophes excluded) and open square
brackets whose content has no digit (so citations and confidence intervals do not count). A marker
is enclosed when one of these is still open at the marker and a matching close follows it.
"""
import argparse
import json
import re
from pathlib import Path

import polars as pl

from normative_discourse.config.settings import PROJECT_ROOT
from normative_discourse.preprocessing.text_normalizer import normalize_for_detection

LEFT_CTX = set(" \t([{\u2014\u2013-/:;\u2026")
RIGHT_CTX = set(" \t.,;:!?)]}\u2014\u2013-/\u2026")


def is_open(t: str, i: int) -> bool:
    prev = t[i - 1] if i > 0 else " "
    nxt = t[i + 1] if i + 1 < len(t) else " "
    return (prev in LEFT_CTX or prev in "\"'\u201c\u2018") and not nxt.isspace()


def is_close(t: str, i: int) -> bool:
    prev = t[i - 1] if i > 0 else " "
    nxt = t[i + 1] if i + 1 < len(t) else " "
    return not prev.isspace() and (nxt in RIGHT_CTX or nxt in "\"'\u201d\u2019")


def state_at(p: str, pos: int) -> dict:
    """Open double, single and bracket marks in p[:pos]."""
    dq = sq = br = False
    for i in range(pos):
        ch = p[i]
        if ch == "\u201c":
            dq = True
        elif ch == "\u201d":
            dq = False
        elif ch == '"':
            if dq and is_close(p, i):
                dq = False
            elif not dq and is_open(p, i):
                dq = True
        elif ch == "\u2018":
            sq = True
        elif ch in "\u2019'":
            if sq and is_close(p, i):
                sq = False
            elif ch == "'" and not sq and is_open(p, i):
                sq = True
        elif ch == "[":
            j = p.find("]", i)
            if j > 0 and not re.search(r"\d", p[i:j]):
                br = True
        elif ch == "]":
            br = False
    return {"double": dq, "single": sq, "bracket": br}


def closes_after(p: str, pos: int, kind: str) -> bool:
    for i in range(pos, len(p)):
        ch = p[i]
        if kind == "double" and (ch == "\u201d" or (ch == '"' and is_close(p, i))):
            return True
        if kind == "single" and ch in "\u2019'" and is_close(p, i):
            return True
        if kind == "bracket" and ch == "]":
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sample.json")
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/processed/articles_corpus_10yr_clean.parquet")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/auto_quoted.json")
    args = ap.parse_args()
    with args.sample.open(encoding="utf-8") as f:
        sample = json.load(f)
    corpus = pl.read_parquet(args.corpus, columns=["pmid", "results"])
    texts = dict(corpus.iter_rows())
    out = {}
    for r in sample:
        t = normalize_for_detection(texts[r["pmid"]] or "")
        s, e = r["start_char"], r["end_char"]
        p0 = t.rfind("\n", 0, s) + 1
        p1 = t.find("\n", e)
        p1 = len(t) if p1 < 0 else p1
        p = t[p0:p1]
        st = state_at(p, s - p0)
        out[r["id"]] = [k for k, v in st.items() if v and closes_after(p, e - p0, k)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(sum(1 for v in out.values() if v), "enclosed of", len(out))


if __name__ == "__main__":
    main()
