"""Tally speaker judgements, exact binomial intervals, and human/Sonnet agreement.

Usage::

    python scripts/quote_check/tally_speaker.py --human path/to/artifact-export
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import beta

from normative_discourse.config.settings import PROJECT_ROOT

CATS_V1 = ["participant", "author", "other_source", "unclear"]
CATS = ["participant", "participant_summary", "author", "other_source", "unclear"]
SONNET_V1_DIR = PROJECT_ROOT / "data/processed/quote_check/sonnet_v1"
FIELDS = ["medical_education", "general_medicine"]
BATCH_SIZE = 20
DEFAULT_LABELS = PROJECT_ROOT / "data/quote_check/speaker_labels.csv"
LABEL_COLUMNS = [
    "id", "pmid", "journal", "journal_field", "expression", "expression_group",
    "start_char", "end_char", "sonnet_v1_speaker", "sonnet_speaker", "human_speaker", "human_auto",
]


def load_labels(path: Path) -> list[dict]:
    """Read only the public schema, rejecting incomplete or ambiguous labels."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LABEL_COLUMNS:
            raise SystemExit("labels CSV must have the exact public header")
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise SystemExit("labels CSV row has the wrong number of columns")
        record_id = row["id"]
        if not record_id or record_id in ids:
            raise SystemExit(f"empty or duplicate labels id: {record_id}")
        ids.add(record_id)
        if row["journal_field"] not in FIELDS:
            raise SystemExit(f"unknown journal field for {record_id}")
        for column, categories in (("sonnet_speaker", CATS), ("human_speaker", ["", *CATS]),
                                   ("sonnet_v1_speaker", ["", *CATS_V1])):
            if row[column] not in categories:
                raise SystemExit(f"invalid {column} category for {record_id}")
        if row["human_auto"] not in ("true", "false"):
            raise SystemExit(f"invalid human_auto for {record_id}")
        try:
            start, end = int(row["start_char"]), int(row["end_char"])
        except ValueError:
            raise SystemExit(f"invalid offsets for {record_id}") from None
        if not 0 <= start < end:
            raise SystemExit(f"invalid offsets for {record_id}")
    if any(r["sonnet_v1_speaker"] for r in rows) and not all(r["sonnet_v1_speaker"] for r in rows):
        raise SystemExit("sonnet_v1_speaker must be complete or entirely empty")
    return rows


def make_labels_report(labels_path: Path, meta_path: Path | None = None,
                       model_rr: float = 16.94) -> str:
    """Reconstruct the upstream tally without distributing passages or notes."""
    sample = load_labels(labels_path)
    meta_path = meta_path or labels_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sonnet = {r["id"]: r["sonnet_speaker"] for r in sample}
    human = {r["id"]: r["human_speaker"] for r in sample if r["human_speaker"]}
    auto_ids = {r["id"] for r in sample if r["human_auto"] == "true"}
    v1 = {r["id"]: r["sonnet_v1_speaker"] for r in sample if r["sonnet_v1_speaker"]}
    first_round = (v1, meta["sonnet_v1_evidence_unverifiable_ids"]) if v1 else None
    provenance = [
        f"Upstream sample SHA-256: `{meta['sample_sha256']}`",
        f"Labels SHA-256: `{hashlib.sha256(labels_path.read_bytes()).hexdigest()}`",
    ]
    return render_report(sample, sonnet, human, auto_ids, meta["sonnet_evidence_unverifiable_ids"],
                         first_round, provenance, model_rr, upstream_evidence=True)


def cp_ci(k: int, n: int) -> tuple[float, float]:
    """Return an exact two-sided Clopper-Pearson 95% interval."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Clopper-Pearson requires n > 0 and 0 <= k <= n")
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def load_sonnet(sample: list[dict], directory: Path,
                categories: list[str] = CATS) -> tuple[dict, list[str]]:
    """Validate complete ordered batches and collect ids with invalid evidence."""
    sonnet = {}
    bad_evidence = []
    expected_files = set()
    for start in range(0, len(sample), BATCH_SIZE):
        number = f"{start // BATCH_SIZE:02d}"
        records = sample[start:start + BATCH_SIZE]
        ids = [r["id"] for r in records]
        batch_path = directory / f"batch_{number}.json"
        if batch_path.exists():
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            if [r["id"] for r in batch] != ids:
                raise SystemExit(f"id order mismatch in batch {number} input")
        path = directory / f"out_{number}.json"
        expected_files.add(path.name)
        if not path.exists():
            raise SystemExit(f"missing Sonnet judgements for ids: {', '.join(ids)}")
        output = json.loads(path.read_text(encoding="utf-8"))
        if [o["id"] for o in output] != ids:
            raise SystemExit(f"id order mismatch in batch {number} output; expected ids: {', '.join(ids)}")
        for record, judgement in zip(records, output):
            record_id = record["id"]
            if record_id in sonnet:
                raise SystemExit(f"duplicate sampled id: {record_id}")
            if judgement.get("speaker") not in categories:
                raise SystemExit(f"invalid category for {record_id}")
            evidence = judgement.get("evidence")
            passage = record["before"] + record["marker"] + record["after"]
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in passage:
                bad_evidence.append(record_id)
            sonnet[record_id] = judgement["speaker"]
    extras = sorted(p.name for p in directory.glob("out_*.json") if p.name not in expected_files)
    if extras:
        raise SystemExit(f"unexpected Sonnet output files: {', '.join(extras)}")
    return sonnet, bad_evidence


def load_human(directory: Path | None, sample_ids: set[str],
               auto_ids: set[str] | None = None) -> dict[str, str]:
    """Read artifact documents and optionally collect ids with an auto: note."""
    human = {}
    if directory is None:
        return human
    if not directory.is_dir():
        raise SystemExit(f"human export directory does not exist: {directory}")
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        data = document.get("data") or {}
        speaker = document.get("speaker") or data.get("speaker")
        if not speaker:
            continue
        record_id = document.get("id")
        if record_id not in sample_ids:
            record_id = path.stem
        if record_id not in sample_ids:
            raise SystemExit(f"unknown human judgement id in {path.name}")
        if speaker not in CATS:
            raise SystemExit(f"invalid human category for {record_id}")
        if record_id in human:
            raise SystemExit(f"duplicate human judgement for {record_id}")
        human[record_id] = speaker
        body = document if document.get("speaker") else data
        note = body.get("note")
        if auto_ids is not None and isinstance(note, str) and note.startswith("auto:"):
            auto_ids.add(record_id)
    return human


def agreement(sonnet: dict, human: dict) -> tuple[int, int, float | None]:
    """Return agreement count, paired count, and five-category Cohen's kappa."""
    ids = [i for i in sonnet if i in human]
    n = len(ids)
    agree = sum(sonnet[i] == human[i] for i in ids)
    if not n:
        return agree, n, None
    sc = Counter(sonnet[i] for i in ids)
    hc = Counter(human[i] for i in ids)
    expected = sum(sc[cat] * hc[cat] for cat in CATS) / (n * n)
    kappa = None if expected == 1 else (agree / n - expected) / (1 - expected)
    return agree, n, kappa


def category_table(ids: list[str], judgements: dict,
                   categories: list[str] = CATS) -> list[str]:
    """Format counts and intervals using only records with a judgement."""
    counts = Counter(judgements[i] for i in ids if i in judgements)
    n = sum(counts.values())
    lines = [
        "| Category | Count | Percent | Exact 95% CI |",
        "| --- | ---: | ---: | --- |",
    ]
    for cat in categories:
        k = counts[cat]
        if n:
            lo, hi = cp_ci(k, n)
            lines.append(f"| {cat} | {k} | {k / n:.1%} | [{lo:.1%}, {hi:.1%}] |")
        else:
            lines.append(f"| {cat} | 0 | N/A | N/A (no judgements) |")
    return lines


def grouped_table(ids: list[str], human: dict) -> list[str]:
    """Combine human categories into the three headline voice groups."""
    groups = {
        "participant": "participant voice",
        "participant_summary": "participant voice",
        "author": "author or document",
        "other_source": "author or document",
        "unclear": "unclear",
    }
    grouped = {i: groups[speaker] for i, speaker in human.items()}
    return category_table(ids, grouped, ["participant voice", "author or document", "unclear"])


def auto_counts(ids: list[str], auto_ids: set[str], sonnet: dict) -> str:
    """Summarize mechanically assigned human documents without their notes."""
    selected = [i for i in ids if i in auto_ids]
    participants = sum(sonnet.get(i) == "participant" for i in selected)
    return (f"Human documents with auto: notes: {len(selected)}; "
            f"Sonnet judged participant among these: {participants}.")


def make_report(sample_path: Path, sonnet_dir: Path, human_dir: Path | None,
                model_rr: float = 16.94, sonnet_v1_dir: Path = SONNET_V1_DIR) -> str:
    """Return a passage-free Markdown report for the supplied frozen sample."""
    if not math.isfinite(model_rr) or model_rr < 0:
        raise SystemExit("model RR must be finite and non-negative")
    raw = sample_path.read_bytes()
    sample = json.loads(raw)
    if any(r["journal_field"] not in FIELDS for r in sample):
        raise SystemExit("unknown journal field in sample")
    sonnet, bad_evidence = load_sonnet(sample, sonnet_dir)
    first_round = (load_sonnet(sample, sonnet_v1_dir, CATS_V1)
                   if sonnet_v1_dir.exists() else None)
    auto_ids = set()
    human = load_human(human_dir, set(sonnet), auto_ids)
    return render_report(sample, sonnet, human, auto_ids, bad_evidence, first_round,
                         [f"Sample SHA-256: `{hashlib.sha256(raw).hexdigest()}`"], model_rr)


def render_report(sample: list[dict], sonnet: dict, human: dict, auto_ids: set[str],
                  bad_evidence: list[str], first_round: tuple | None, provenance: list[str],
                  model_rr: float, upstream_evidence: bool = False) -> str:
    """Format shared counts and statistics for either input mode."""
    if not math.isfinite(model_rr) or model_rr < 0:
        raise SystemExit("model RR must be finite and non-negative")
    evidence_suffix = (" (checked upstream; evidence spans are not distributed)"
                       if upstream_evidence else "")
    articles = Counter(r["pmid"] for r in sample)
    lines = [
        "# Results marker speaker check",
        "",
        *provenance,
        "",
        f"Sampled markers: {len(sample)}; distinct articles: {len(articles)}; "
        f"articles with more than one sampled marker: {sum(n > 1 for n in articles.values())}.",
        "",
    ]
    field_ids = {field: [r["id"] for r in sample if r["journal_field"] == field] for field in FIELDS}
    for field, ids in field_ids.items():
        lines.append(f"Sample n, {field}: {len(ids)}.")
    lines.extend([
        "",
        f"Sonnet judgements: {len(sonnet)}/{len(sample)}.",
        "",
        "Evidence empty or not verbatim (ids): " + (", ".join(bad_evidence) or "none")
        + evidence_suffix + ".",
        "",
        f"Human judgements missing: {len(sample) - len(human)}; human percentages use judged records only.",
        "",
        auto_counts([r["id"] for r in sample], auto_ids, sonnet),
    ])
    share_groups = [
        ("participant (quotations)", {"participant"}),
        ("participant or participant_summary", {"participant", "participant_summary"}),
    ]
    shares = {label: {} for label, _ in share_groups}
    for field, ids in field_ids.items():
        judged = sum(i in human for i in ids)
        lines.extend([
            "", f"## {field} (sample n={len(ids)})", "",
            f"Human (primary): judged n={judged}; missing {len(ids) - judged}.", "",
            *grouped_table(ids, human), "",
            auto_counts(ids, auto_ids, sonnet), "",
            "Human categories:", "", *category_table(ids, human), "",
            f"Sonnet: n={len(ids)}.", "",
            *category_table(ids, sonnet),
        ])
        lines.extend(["", "Human participant shares:", "",
                      "| Share | Count | Percent | Exact 95% CI |",
                      "| --- | ---: | ---: | --- |"])
        for label, categories in share_groups:
            k = sum(human.get(i) in categories for i in ids)
            shares[label][field] = k / judged if judged else None
            if judged:
                lo, hi = cp_ci(k, judged)
                lines.append(f"| {label} | {k} | {k / judged:.1%} | [{lo:.1%}, {hi:.1%}] |")
            else:
                lines.append(f"| {label} | 0 | N/A | N/A (no judgements) |")
    if first_round is not None:
        v1, bad_v1 = first_round
        lines.extend(["", "## Sonnet, first round (four categories)", "",
                      f"Sonnet judgements: {len(v1)}/{len(sample)}.", "",
                      "Evidence empty or not verbatim (ids): " + (", ".join(bad_v1) or "none")
                      + evidence_suffix + "."])
        for field, ids in field_ids.items():
            lines.extend(["", f"### {field} (n={len(ids)})", "",
                          *category_table(ids, v1, CATS_V1)])
    agree, n, kappa = agreement(sonnet, human)
    raw_agreement = f"{agree}/{n} ({agree / n:.1%})" if n else "N/A (no paired judgements)"
    kappa_text = f"{kappa:.6f}" if kappa is not None else "N/A (undefined)"
    lines.extend([
        "", "## Agreement", "",
        f"Raw agreement: {raw_agreement}; Cohen's kappa (five categories): {kappa_text}.", "",
        "The human saw Sonnet's draft before deciding, so this is not an independent agreement.", "",
        "Disagreements (id, Sonnet, human):", "",
    ])
    disagreements = [i for i in sonnet if i in human and sonnet[i] != human[i]]
    lines.extend(f"- {i}, {sonnet[i]}, {human[i]}" for i in disagreements)
    if not disagreements:
        lines.append("None.")
    lines.extend(["", "## Crude scaling", "", f"Model Results RR excluding combined-section articles: {model_rr:g}.", ""])
    formula = "model RR × (1 − p_removed,ME) / (1 − p_removed,GM)"
    for label, categories in share_groups:
        me = shares[label]["medical_education"]
        gm = shares[label]["general_medicine"]
        removed = " + ".join(cat for cat in CATS if cat in categories)
        if me is None or gm is None:
            scaled = "N/A (human judgements required in both fields)"
        elif gm == 1:
            scaled = f"N/A (zero denominator: all judged GM markers are {label})"
        else:
            scaled = f"{model_rr * (1 - me) / (1 - gm):.4f} ({model_rr:g} × (1 − {me:.6f}) / (1 − {gm:.6f}))"
        lines.extend([
            f"Removing {removed}: {formula} = {scaled}; "
            "crude approximation, not a re-estimated rate ratio.", "",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--labels-meta", type=Path)
    ap.add_argument("--sonnet-dir", type=Path, default=PROJECT_ROOT / "data/processed/quote_check/sonnet")
    ap.add_argument("--sonnet-v1-dir", type=Path, default=SONNET_V1_DIR)
    ap.add_argument("--human", type=Path)
    ap.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/quote_check/summary.md")
    ap.add_argument("--model-rr", type=float, default=16.94)
    args = ap.parse_args()
    if args.sample is not None and args.labels is not None:
        ap.error("--sample and --labels are mutually exclusive")
    labels = args.labels
    if args.sample is None and labels is None and DEFAULT_LABELS.exists():
        labels = DEFAULT_LABELS
    if labels is not None:
        report = make_labels_report(labels, args.labels_meta, args.model_rr)
    else:
        sample = args.sample or PROJECT_ROOT / "data/processed/quote_check/sample.json"
        report = make_report(sample, args.sonnet_dir, args.human, args.model_rr, args.sonnet_v1_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
