"""Synthetic tests for the reproducible Results speaker check workflow."""
from __future__ import annotations

import hashlib
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest
from scipy.stats import beta

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "quote_check"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampler = load_script("sample_results_markers")
batches = load_script("make_batches")
review = load_script("build_review_page")
tally = load_script("tally_speaker")
auto_quoted = load_script("auto_quoted")
exporter = load_script("export_labels")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def run_main(module, monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, arguments)])
    module.main()


@pytest.fixture
def sampling_inputs(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    matches = tmp_path / "matches.parquet"
    combined = tmp_path / "combined.csv"
    articles = []
    markers = []
    for index in range(8):
        pmid = str(index)
        articles.append({
            "pmid": pmid,
            "journal": "Synthetic Journal",
            "journal_field": tally.FIELDS[index // 4],
            "is_non_imrad": True if index % 4 == 3 else (None if index % 4 == 1 else False),
            "results": "“" + "must " * 300 + "é",
        })
        for start in (11, 606, 1201):
            markers.append({
                "pmid": pmid,
                "section": "results",
                "start_char": start,
                "end_char": start + 4,
                "expression": "must",
                "expression_group": "obligation",
            })
        markers.append({**markers[-1], "section": "discussion"})
    pl.DataFrame(articles).write_parquet(corpus)
    pl.DataFrame(markers).write_parquet(matches)
    pl.DataFrame({"pmid": ["2", "6"]}).write_csv(combined)
    return corpus, matches, combined


def test_sampler_selection_serialization_and_freeze(sampling_inputs, tmp_path, monkeypatch):
    corpus, matches, combined = sampling_inputs
    out = tmp_path / "frozen" / "sample.json"
    arguments = ["--corpus", corpus, "--matches", matches, "--combined", combined,
                 "--out", out, "--n-per-field", "4"]
    run_main(sampler, monkeypatch, *arguments)
    original = out.read_bytes()
    sample = json.loads(original)
    assert len(sample) == 8
    assert [r["id"] for r in sample] == [f"q{i:03d}" for i in range(8)]
    assert [(r["pmid"], r["start_char"]) for r in sample] == [
        ("1", 1201), ("5", 11), ("4", 11), ("0", 1201),
        ("5", 1201), ("0", 11), ("5", 606), ("0", 606),
    ]
    assert {r["pmid"] for r in sample} == {"0", "1", "4", "5"}
    assert all(sum(r["journal_field"] == f for r in sample) == 4 for f in tally.FIELDS)
    assert list(sample[0]) == [
        "id", "pmid", "journal", "journal_field", "expression", "expression_group",
        "start_char", "end_char", "before", "marker", "after",
    ]
    assert sampler.SEED == 20261018
    assert sampler.WINDOW == 600
    for record in sample:
        assert record["marker"] == "must"
        assert len(record["before"]) == min(600, record["start_char"])
        assert len(record["after"]) == min(600, 1502 - record["end_char"])
    expected = json.dumps(sample, ensure_ascii=False, indent=1).replace("\n", os.linesep)
    assert original == expected.encode("utf-8")
    with pytest.raises(SystemExit, match="frozen"):
        run_main(sampler, monkeypatch, *arguments)
    assert out.read_bytes() == original
    second = tmp_path / "second.json"
    arguments[arguments.index(out)] = second
    run_main(sampler, monkeypatch, *arguments)
    assert second.read_bytes() == original


def records(n):
    return [{
        "id": f"q{i:03d}", "pmid": str(i // 2), "journal": "Synthetic",
        "journal_field": tally.FIELDS[0 if i < 2 else 1],
        "before": "PRIVATE PASSAGE é </script> said ", "marker": "must",
        "after": " PRIVATE CONTINUATION",
    } for i in range(n)]


def test_batches_and_prompts(tmp_path, monkeypatch):
    sample = records(21)
    path = tmp_path / "sample.json"
    write_json(path, sample)
    directory = tmp_path / "sonnet"
    run_main(batches, monkeypatch, "--sample", path, "--out-dir", directory)
    assert sorted(p.name for p in directory.iterdir()) == [
        "batch_00.json", "batch_01.json", "prompt_00.md", "prompt_01.md",
    ]
    for i, expected in enumerate((sample[:20], sample[20:])):
        batch = json.loads((directory / f"batch_{i:02d}.json").read_text(encoding="utf-8"))
        assert batch == [{k: r[k] for k in ("id", "before", "marker", "after")} for r in expected]
        prompt = (directory / f"prompt_{i:02d}.md").read_text(encoding="utf-8")
        assert "{DIR}" not in prompt
        assert "NN" not in prompt
        assert f"{directory.resolve().as_posix()}/batch_{i:02d}.json" in prompt
        assert f"{directory.resolve().as_posix()}/out_{i:02d}.json" in prompt
        assert f"{len(expected)} records" in prompt
        assert f"{len(expected)} objects" in prompt


@pytest.fixture
def judged_inputs(tmp_path):
    sample = records(4)
    path = tmp_path / "sample.json"
    write_json(path, sample)
    directory = tmp_path / "sonnet"
    batches.make_batches(path, directory)
    output = [{
        "id": r["id"], "speaker": cat,
        "reason": "PRIVATE REASON", "evidence": "said must",
    } for r, cat in zip(sample, tally.CATS_V1)]
    write_json(directory / "out_00.json", output)
    human = tmp_path / "human"
    human.mkdir()
    write_json(human / "q000.json", {"speaker": "participant", "note": "PRIVATE NOTE"})
    write_json(human / "exported.json", {"id": "q001", "data": {"speaker": "author"}})
    write_json(human / "q002.json", {"id": "unrelated-document-id", "data": {"speaker": "unclear"}})
    write_json(human / "fourth.json", {"id": "q003", "speaker": "other_source"})
    return path, directory, human


def test_review_records_and_script_escaping(judged_inputs, tmp_path, monkeypatch):
    path, directory, _ = judged_inputs
    out = tmp_path / "review.html"
    run_main(review, monkeypatch, "--sample", path, "--sonnet-dir", directory, "--out", out)
    html = out.read_text(encoding="utf-8")
    assert "__RECORDS__" not in html
    payload = html.split('<script id="records" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "</" not in payload
    assert "<\\/script>" in payload
    loaded = json.loads(payload)
    assert [r["id"] for r in loaded] == [r["id"] for r in records(4)]
    assert list(loaded[0]) == ["id", "before", "marker", "after", "sonnet"]
    assert loaded[0]["sonnet"] == {
        "speaker": "participant", "reason": "PRIVATE REASON", "evidence": "said must",
    }
    assert loaded[0]["before"] == records(4)[0]["before"]


def test_tally_report_counts_ci_kappa_and_exports(judged_inputs, tmp_path, monkeypatch, capsys):
    path, directory, human = judged_inputs
    assert tally.cp_ci(1, 2) == pytest.approx((beta.ppf(0.025, 1, 2), beta.ppf(0.975, 2, 1)))
    sonnet, bad = tally.load_sonnet(records(4), directory)
    assert bad == []
    judgements = tally.load_human(human, set(sonnet))
    assert judgements == dict(zip([f"q{i:03d}" for i in range(4)], ["participant", "author", "unclear", "other_source"]))
    agree, n, kappa = tally.agreement(sonnet, judgements)
    assert (agree, n) == (2, 4)
    assert kappa == pytest.approx(1 / 3)
    out = tmp_path / "report" / "summary.md"
    run_main(tally, monkeypatch, "--sample", path, "--sonnet-dir", directory,
             "--human", human, "--report", out)
    report = out.read_text(encoding="utf-8")
    assert capsys.readouterr().out == report
    assert hashlib.sha256(path.read_bytes()).hexdigest() in report
    assert "distinct articles: 2; articles with more than one sampled marker: 2" in report
    for field in tally.FIELDS:
        assert f"Sample n, {field}: 2" in report
    lo, hi = tally.cp_ci(1, 2)
    assert f"| participant | 1 | 50.0% | [{lo:.1%}, {hi:.1%}] |" in report
    assert "2/4 (50.0%)" in report
    assert "0.333333" in report
    assert "Cohen's kappa (five categories)" in report
    assert "not an independent agreement" in report
    assert "- q002, other_source, unclear" in report
    assert "- q003, unclear, other_source" in report
    assert "8.4700" in report
    assert "not a re-estimated rate ratio" in report
    assert "PRIVATE" not in report
    assert "said must" not in report


@pytest.mark.parametrize("evidence", ["", "invented span", "   "])
def test_invalid_evidence_reports_ids(judged_inputs, evidence):
    path, directory, human = judged_inputs
    output_path = directory / "out_00.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output[1]["evidence"] = evidence
    write_json(output_path, output)
    report = tally.make_report(path, directory, human)
    assert "Evidence empty or not verbatim (ids): q001." in report
    assert "invented span" not in report


@pytest.mark.parametrize("first_round", [False, True])
@pytest.mark.parametrize("problem", ["order", "input_order", "category", "missing", "short", "extra"])
def test_sonnet_validation(judged_inputs, problem, first_round, tmp_path):
    path, directory, human = judged_inputs
    output_path = directory / "out_00.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    if problem == "order":
        write_json(output_path, list(reversed(output)))
    elif problem == "input_order":
        write_json(directory / "batch_00.json", list(reversed(records(4))))
    elif problem == "category":
        output[0]["speaker"] = "invalid"
        write_json(output_path, output)
    elif problem == "missing":
        output_path.unlink()
    elif problem == "short":
        write_json(output_path, output[:-1])
    else:
        write_json(directory / "out_01.json", [])
    with pytest.raises(SystemExit):
        if first_round:
            tally.make_report(path, directory, human, sonnet_v1_dir=directory)
        else:
            tally.make_report(path, directory, human, sonnet_v1_dir=tmp_path / "absent")


def test_missing_human_and_undefined_statistics(judged_inputs):
    path, directory, human = judged_inputs
    write_json(human / "unjudged.json", {"id": "q003", "data": {"speaker": "", "note": "note only"}})
    (human / "fourth.json").unlink()
    report = tally.make_report(path, directory, human, 10)
    assert "Human judgements missing: 1" in report
    assert "Human (primary): judged n=1; missing 1" in report
    assert "2/3 (66.7%)" in report
    report = tally.make_report(path, directory, None)
    assert "Human judgements missing: 4" in report
    assert "N/A (no judgements)" in report
    assert "N/A (no paired judgements)" in report
    assert "N/A (human judgements required in both fields)" in report
    write_json(human / "q002.json", {"speaker": "participant"})
    report = tally.make_report(path, directory, human)
    assert "N/A (zero denominator" in report
    assert tally.agreement({"x": "author"}, {"x": "author"}) == (1, 1, None)
    assert tally.cp_ci(0, 2)[0] == 0
    assert tally.cp_ci(2, 2)[1] == 1
    with pytest.raises(ValueError):
        tally.cp_ci(0, 0)


@pytest.mark.parametrize("problem", ["category", "duplicate", "unknown"])
def test_invalid_human_exports(judged_inputs, problem):
    path, directory, human = judged_inputs
    if problem == "category":
        document = {"id": "q000", "data": {"speaker": "invalid"}}
        target = human / "q000.json"
    elif problem == "duplicate":
        document = {"id": "q000", "speaker": "author"}
        target = human / "duplicate.json"
    else:
        document = {"id": "absent", "speaker": "author"}
        target = human / "unknown.json"
    write_json(target, document)
    with pytest.raises(SystemExit):
        tally.make_report(path, directory, human)


@pytest.mark.parametrize("category", ["participant", "participant_summary", "author", "other_source", "unclear"])
def test_five_category_validation(judged_inputs, category):
    path, directory, human = judged_inputs
    assert tally.CATS == ["participant", "participant_summary", "author", "other_source", "unclear"]
    output_path = directory / "out_00.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output[0]["speaker"] = category
    write_json(output_path, output)
    write_json(human / "q000.json", {"speaker": category})
    assert tally.load_sonnet(records(4), directory)[0]["q000"] == category
    assert tally.load_human(human, {r["id"] for r in records(4)})["q000"] == category


def test_five_category_kappa():
    sonnet = dict(zip("abcdef", [*tally.CATS, "participant"]))
    human = dict(zip("abcdef", ["participant_summary", "participant", *tally.CATS[2:], "participant"]))
    # Marginals are (2,1,1,1,1), so expected = 8/36 and kappa = 4/7.
    assert tally.agreement(sonnet, human) == pytest.approx((4, 6, 4 / 7))


def test_both_human_shares_and_scalings(judged_inputs, tmp_path):
    path, directory, human = judged_inputs
    write_json(human / "exported.json", {"id": "q001", "speaker": "participant_summary"})
    report = tally.make_report(path, directory, human, 10, tmp_path / "absent")
    me, gm = report.split("## medical_education", 1)[1].split("## general_medicine", 1)
    for section, k_quote, k_combined in [(me, 1, 2), (gm, 0, 0)]:
        for label, k in [("participant (quotations)", k_quote),
                         ("participant or participant_summary", k_combined)]:
            lo, hi = tally.cp_ci(k, 2)
            assert f"| {label} | {k} | {k / 2:.1%} | [{lo:.1%}, {hi:.1%}] |" in section
    lines = [line for line in report.splitlines() if line.startswith("Removing ")]
    assert len(lines) == 2
    formula = "model RR × (1 − p_removed,ME) / (1 − p_removed,GM)"
    assert lines[0].startswith(f"Removing participant: {formula} = 5.0000")
    assert lines[1].startswith(f"Removing participant + participant_summary: {formula} = 0.0000")
    assert all("crude approximation, not a re-estimated rate ratio" in line for line in lines)


def test_optional_first_round_cli(judged_inputs, tmp_path, monkeypatch):
    path, directory, human = judged_inputs
    v1 = tmp_path / "sonnet_v1"
    report_path = tmp_path / "report.md"
    args = ["--sample", path, "--sonnet-dir", directory, "--human", human,
            "--sonnet-v1-dir", v1, "--report", report_path]
    run_main(tally, monkeypatch, *args)
    assert "Sonnet, first round" not in report_path.read_text(encoding="utf-8")
    v1.mkdir()
    output = [{"id": r["id"], "speaker": cat, "evidence": "said must"}
              for r, cat in zip(records(4), tally.CATS_V1)]
    output[0]["evidence"] = "invented"
    write_json(v1 / "out_00.json", output)
    run_main(tally, monkeypatch, *args)
    section = report_path.read_text(encoding="utf-8").split(
        "## Sonnet, first round (four categories)", 1)[1].split("## Agreement", 1)[0]
    assert "Evidence empty or not verbatim (ids): q000." in section
    lo, hi = tally.cp_ci(1, 2)
    for field, cats in zip(tally.FIELDS, [tally.CATS_V1[:2], tally.CATS_V1[2:]]):
        field_section = section.split(f"### {field} (n=2)", 1)[1].split("###", 1)[0]
        for cat in cats:
            assert f"| {cat} | 1 | 50.0% | [{lo:.1%}, {hi:.1%}] |" in field_section
    assert "participant_summary" not in section
    output[0]["speaker"] = "participant_summary"
    write_json(v1 / "out_00.json", output)
    with pytest.raises(SystemExit, match="invalid category"):
        run_main(tally, monkeypatch, *args)


def test_legacy_human_values_are_accepted(judged_inputs):
    _, _, human = judged_inputs
    loaded = tally.load_human(human, {r["id"] for r in records(4)})
    assert set(loaded.values()) == set(tally.CATS_V1)


@pytest.mark.parametrize("paragraph, kinds", [
    ('They said “we must act”.', ["double"]),
    ('They said "we must act".', ["double"]),
    ("God's view: 'we didn't think we must act' today.", ["single"]),
    ('They said "we must act"… then left.', ["double"]),
    ('They said ‘we must act’ today.', ["single"]),
    ('The range [we must act in 2 days] applies.', []),
    ('The note [we must act soon] applies.', ["bracket"]),
    ('They said "we must act without closure.', []),
    ('They said "finished" and must act.', []),
    ("God's view didn't mean we must act.", []),
])
def test_auto_quoted_paragraph_state_and_closure(paragraph, kinds):
    start = paragraph.index("must")
    end = start + len("must")
    state = auto_quoted.state_at(paragraph, start)
    assert [k for k, opened in state.items()
            if opened and auto_quoted.closes_after(paragraph, end, k)] == kinds
    for kind in kinds:
        assert state[kind]
        assert auto_quoted.closes_after(paragraph, end, kind)
    assert auto_quoted.state_at(paragraph, 0) == dict.fromkeys(state, False)
    assert not any(auto_quoted.closes_after(paragraph, len(paragraph), k) for k in state)


def test_auto_quoted_cli_normalization_and_whole_paragraph(tmp_path, monkeypatch, capsys):
    text = '  “' + 'words ' * 120 + 'must' + ' words' * 120 + '”  '
    normalized = sampler.normalize_for_detection(text)
    start = normalized.index("must")
    sample = [{"id": "q000", "pmid": "1", "start_char": start, "end_char": start + 4},
              {"id": "q001", "pmid": "2", "start_char": 3, "end_char": 7}]
    sample_path = tmp_path / "sample.json"
    corpus = tmp_path / "corpus.parquet"
    out = tmp_path / "output" / "auto_quoted.json"
    write_json(sample_path, sample)
    pl.DataFrame({"pmid": ["1", "2"], "results": [text, "We must act."]}).write_parquet(corpus)
    run_main(auto_quoted, monkeypatch, "--sample", sample_path, "--corpus", corpus, "--out", out)
    expected = {"q000": ["double"], "q001": []}
    assert json.loads(out.read_text(encoding="utf-8")) == expected
    assert out.read_text(encoding="utf-8") == json.dumps(expected, ensure_ascii=False, indent=1)
    assert capsys.readouterr().out == "1 enclosed of 2\n"


def test_grouped_headline_and_auto_counts(judged_inputs, tmp_path):
    path, directory, human = judged_inputs
    write_json(human / "q000.json", {"speaker": "participant", "note": "auto: double"})
    write_json(human / "exported.json", {
        "id": "q001", "note": "manual", "data": {"speaker": "participant_summary", "note": "auto: single"},
    })
    write_json(human / "q002.json", {
        "speaker": "author", "note": "manual auto: mention", "data": {"note": "auto: ignored"},
    })
    write_json(human / "fourth.json", {"id": "q003", "speaker": "other_source", "note": "auto: bracket"})
    report = tally.make_report(path, directory, human, sonnet_v1_dir=tmp_path / "absent")
    assert "Human documents with auto: notes: 3; Sonnet judged participant among these: 1." in report
    me, gm = report.split("## medical_education", 1)[1].split("## general_medicine", 1)
    for section, counts, auto_n, sonnet_n in [(me, [2, 0, 0], 2, 1), (gm, [0, 2, 0], 1, 0)]:
        assert section.index("| participant voice |") < section.index("Human categories:")
        assert section.index("| participant voice |") < section.index("| participant |")
        for label, k in zip(["participant voice", "author or document", "unclear"], counts):
            lo, hi = tally.cp_ci(k, 2)
            assert f"| {label} | {k} | {k / 2:.1%} | [{lo:.1%}, {hi:.1%}] |" in section
        assert (f"Human documents with auto: notes: {auto_n}; "
                f"Sonnet judged participant among these: {sonnet_n}.") in section
    (human / "fourth.json").unlink()
    report = tally.make_report(path, directory, human, sonnet_v1_dir=tmp_path / "absent")
    gm = report.split("## general_medicine", 1)[1]
    lo, hi = tally.cp_ci(1, 1)
    assert f"| author or document | 1 | 100.0% | [{lo:.1%}, {hi:.1%}] |" in gm
    report = tally.make_report(path, directory, None, sonnet_v1_dir=tmp_path / "absent")
    assert "| participant voice | 0 | N/A | N/A (no judgements) |" in report
    assert report.count("Human documents with auto: notes: 0; Sonnet judged participant among these: 0.") == 3


@pytest.fixture
def exported_labels(judged_inputs, tmp_path):
    path, directory, human = judged_inputs
    sample = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(sample):
        row.update(expression="must", expression_group="must", start_char=i + 1, end_char=i + 5)
    write_json(path, sample)
    write_json(human / "q000.json", {"speaker": "participant", "note": "auto: PRIVATE NOTE"})
    output = json.loads((directory / "out_00.json").read_text(encoding="utf-8"))
    output[1]["evidence"] = "PRIVATE INVALID EVIDENCE"
    write_json(directory / "out_00.json", output)
    out = tmp_path / "public" / "speaker_labels.csv"
    exporter.export_labels(path, directory, directory, human, out)
    return out, path, directory, human


def normalize_report(report):
    return [line for line in report.splitlines() if not (
        "SHA-256" in line or line.startswith("Evidence empty or not verbatim (ids):")
    )]


def test_export_round_trip_and_public_schema(exported_labels, monkeypatch, tmp_path):
    out, path, directory, human = exported_labels
    with out.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "id", "pmid", "journal", "journal_field", "expression", "expression_group",
            "start_char", "end_char", "sonnet_v1_speaker", "sonnet_speaker", "human_speaker", "human_auto",
        ]
        rows = list(reader)
    assert [r["id"] for r in rows] == [r["id"] for r in json.loads(path.read_bytes())]
    assert [r["human_auto"] for r in rows] == ["true", "false", "false", "false"]
    meta = json.loads(out.with_suffix(".meta.json").read_bytes())
    assert meta == {
        "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_sha256_lf": tally.canonical_sha256(path.read_bytes()),
        "seed": 20261018, "n_per_field": 100, "window": 600,
        "sonnet_evidence_unverifiable_ids": ["q001"],
        "sonnet_v1_evidence_unverifiable_ids": ["q001"],
    }
    assert list(meta)[:2] == ["sample_sha256", "sample_sha256_lf"]
    for public_path in (out, out.with_suffix(".meta.json")):
        assert "PRIVATE" not in public_path.read_text(encoding="utf-8")
    sample_report = tally.make_report(path, directory, human, sonnet_v1_dir=directory)
    labels_report = tally.make_labels_report(out)
    assert normalize_report(labels_report) == normalize_report(sample_report)
    assert labels_report.splitlines()[2:5] == [
        f"Upstream sample SHA-256: `{meta['sample_sha256']}`",
        f"Upstream sample SHA-256 (LF-normalized): `{meta['sample_sha256_lf']}`",
        f"Labels SHA-256: `{hashlib.sha256(out.read_bytes()).hexdigest()}`",
    ]
    evidence_line = ("Evidence empty or not verbatim (ids): q001 "
                     "(checked upstream; evidence spans are not distributed).")
    assert labels_report.count(evidence_line) == 2
    report_path = tmp_path / "report.md"
    run_main(tally, monkeypatch, "--labels", out, "--report", report_path)
    assert report_path.read_text(encoding="utf-8") == labels_report
    monkeypatch.setattr(tally, "DEFAULT_LABELS", out)
    run_main(tally, monkeypatch, "--report", report_path)
    assert report_path.read_text(encoding="utf-8") == labels_report
    alternate_meta = tmp_path / "alternate.json"
    write_json(alternate_meta, meta)
    run_main(tally, monkeypatch, "--labels", out, "--labels-meta", alternate_meta, "--report", report_path)
    assert report_path.read_text(encoding="utf-8") == labels_report


@pytest.mark.parametrize("existing", ["csv", "meta", "both"])
def test_export_refuses_either_existing_output(exported_labels, existing, monkeypatch):
    out, path, directory, human = exported_labels
    meta = out.with_suffix(".meta.json")
    if existing == "csv":
        meta.unlink()
    elif existing == "meta":
        out.unlink()
    originals = {p: p.read_bytes() for p in (out, meta) if p.exists()}
    args = ["--sample", path, "--sonnet-dir", directory, "--sonnet-v1-dir", directory,
            "--human", human, "--out", out]
    with pytest.raises(SystemExit, match="exists"):
        run_main(exporter, monkeypatch, *args)
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    run_main(exporter, monkeypatch, *args, "--overwrite")
    assert len(tally.load_labels(out)) == 4
    assert meta.exists()


def rewrite_labels(out, rows, columns=None):
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tally.LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("problem", [
    "header", "extra_header", "duplicate", "empty_id", "field", "sonnet", "human", "v1", "partial_v1",
    "auto", "offset", "negative_offset", "short_row", "long_row",
])
def test_labels_validation(exported_labels, problem):
    out, _, _, _ = exported_labels
    rows = tally.load_labels(out)
    columns = None
    if problem == "header":
        columns = list(reversed(tally.LABEL_COLUMNS))
    elif problem == "extra_header":
        columns = [*tally.LABEL_COLUMNS, "before"]
    elif problem == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "empty_id":
        rows[0]["id"] = ""
    else:
        column, value = {
            "field": ("journal_field", "unknown"), "sonnet": ("sonnet_speaker", ""),
            "human": ("human_speaker", "invalid"), "v1": ("sonnet_v1_speaker", "participant_summary"),
            "partial_v1": ("sonnet_v1_speaker", ""), "auto": ("human_auto", "True"),
            "offset": ("start_char", "text"), "negative_offset": ("start_char", "-1"),
            "short_row": ("human_auto", "false"), "long_row": ("human_auto", "false"),
        }[problem]
        rows[0][column] = value
    rewrite_labels(out, rows, columns)
    if problem in ("short_row", "long_row"):
        lines = out.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].rsplit(",", 1)[0] if problem == "short_row" else lines[1] + ",extra"
        out.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(SystemExit):
        tally.make_labels_report(out)


def test_labels_optional_v1_and_missing_human(exported_labels, tmp_path):
    out, path, directory, human = exported_labels
    rows = tally.load_labels(out)
    for row in rows:
        row["sonnet_v1_speaker"] = ""
    rows[3]["human_speaker"] = ""
    rewrite_labels(out, rows)
    (human / "fourth.json").unlink()
    labels_report = tally.make_labels_report(out)
    report = tally.make_report(path, directory, human, sonnet_v1_dir=tmp_path / "absent")
    assert normalize_report(labels_report) == normalize_report(report)
    assert "Sonnet, first round" not in labels_report
    assert "Human judgements missing: 1" in labels_report


def test_redraw_identity_and_hash_checks(exported_labels, monkeypatch):
    import runpy

    wrapper = runpy.run_path(str(SCRIPT_DIR.parents[1] / "reproducibility/run_08_speaker_check.py"))
    check = wrapper["check_redraw"]
    out, path, _, _ = exported_labels
    assert check(path, out)
    sample = json.loads(path.read_bytes())
    sample[0]["expression"] = "should"
    write_json(path, sample)
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_bytes())
    meta["sample_sha256_lf"] = tally.canonical_sha256(path.read_bytes())
    write_json(meta_path, meta)
    assert not check(path, out)
    sample[0]["expression"] = "must"
    write_json(path, sample)
    assert not check(path, out)


def test_redraw_canonical_hash_across_line_endings(exported_labels, capsys):
    import runpy

    wrapper = runpy.run_path(str(SCRIPT_DIR.parents[1] / "reproducibility/run_08_speaker_check.py"))
    check = wrapper["check_redraw"]
    out, path, _, _ = exported_labels
    lf = path.read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf != crlf
    assert tally.canonical_sha256(lf) == tally.canonical_sha256(crlf)
    assert tally.canonical_sha256(lf) == hashlib.sha256(lf).hexdigest()
    for raw in (crlf, lf):
        path.write_bytes(raw)
        assert check(path, out)
        assert "[PASS] redrawn sample SHA-256 (LF-normalized)" in capsys.readouterr().out
    sample = json.loads(lf)
    sample[0]["before"] += "changed passage"
    write_json(path, sample)
    assert not check(path, out)
    output = capsys.readouterr().out
    assert "[PASS] redrawn sample identifiers and offsets" in output
    assert "[FAIL] redrawn sample SHA-256 (LF-normalized)" in output


def test_missing_canonical_hash_fails_clearly(exported_labels, capsys):
    import runpy

    out, path, _, _ = exported_labels
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_bytes())
    del meta["sample_sha256_lf"]
    write_json(meta_path, meta)
    with pytest.raises(SystemExit, match="labels metadata missing sample_sha256_lf"):
        tally.make_labels_report(out)
    wrapper = runpy.run_path(str(SCRIPT_DIR.parents[1] / "reproducibility/run_08_speaker_check.py"))
    assert not wrapper["check_redraw"](path, out)
    assert "[FAIL] labels metadata missing sample_sha256_lf" in capsys.readouterr().out


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("redraw_ok", [False, True])
def test_speaker_wrapper_tally_skip_and_redraw(tmp_path, monkeypatch, available, redraw_ok, capsys):
    import runpy

    wrapper = runpy.run_path(str(SCRIPT_DIR.parents[1] / "reproducibility/run_08_speaker_check.py"))
    namespace = wrapper["main"].__globals__
    calls = []
    for key in ("CLEAN_PARQUET", "MATCHES_PARQUET", "COMBINED_CSV"):
        path = tmp_path / key
        if available:
            path.touch()
        monkeypatch.setitem(namespace, key, path)
    work = tmp_path / "work/quote_check"
    work.mkdir(parents=True)
    sample = work / "sample.json"
    sample.write_text("stale", encoding="utf-8")
    monkeypatch.setitem(namespace, "HERE", tmp_path)
    monkeypatch.setitem(namespace, "SPEAKER_LABELS", tmp_path / "speaker_labels.csv")

    def fake_run(args, **kwargs):
        calls.append(args)
        if "--out" in args:
            assert not sample.exists()
            assert Path(args[args.index("--out") + 1]) == sample

    monkeypatch.setitem(namespace, "run", fake_run)
    monkeypatch.setitem(namespace, "check_redraw", lambda *args: redraw_ok)
    assert wrapper["main"]() == (0 if not available or redraw_ok else 1)
    assert "--labels" in calls[0]
    assert len(calls) == (2 if available else 1)
    assert ("Redraw check skipped:" in capsys.readouterr().out) == (not available)
    assert sample.exists() == (not available)
