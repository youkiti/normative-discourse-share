"""Tests for the combined Results article exclusion list."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "list_combined_results_articles.py"
_spec = importlib.util.spec_from_file_location("list_combined_results_articles", _SCRIPT_PATH)
combined = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combined)


@pytest.fixture
def inputs(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    corpus_path = tmp_path / "corpus.parquet"
    inventory_path = tmp_path / "inventory.parquet"
    pl.DataFrame({
        "pmid": ["5", "4", "3", "2", "1"],
        "journal": ["ME Journal", "GM Journal", "ME Journal", "ME Journal", "GM Journal"],
        "journal_field": [
            "medical_education", "general_medicine", "medical_education",
            "medical_education", "general_medicine",
        ],
        "is_non_imrad": [False, False, True, None, False],
    }).write_parquet(corpus_path)
    inventory_rows = []
    for pmid, results, discussion, conclusion in (
        ("1", 100, 50, 0),
        ("2", 100, 0, 0),
        ("3", 100, 0, 0),
        ("4", 100, 0, 20),
        ("5", 100, 0, 0),
    ):
        for section, words in (
            ("results", results), ("discussion", discussion), ("conclusion", conclusion)
        ):
            inventory_rows.append({"pmid": pmid, "section": section, "word_count": words})
    pl.DataFrame(inventory_rows).write_parquet(inventory_path)
    titles = {
        "1": ["Results AND Discussion", "Findings & Discussion"],
        "2": ["Discussion & findings"],
        "3": ["Results and Discussion"],
        "4": ["Results", "Conclusion"],
    }
    for pmid, headings in titles.items():
        payload = {
            "sections": [
                {"section_title": title, "section_type": "results"} for title in headings
            ]
        }
        (cleaned / f"{pmid}.json").write_text(json.dumps(payload), encoding="utf-8")
    return cleaned, corpus_path, inventory_path


def test_rules_and_output_columns(inputs, tmp_path):
    articles, unreadable = combined.list_combined_articles(*inputs)
    assert articles.columns == ["pmid", "journal", "journal_field", "reason", "heading"]
    assert articles["pmid"].to_list() == ["1", "2", "5"]
    assert articles["reason"].to_list() == ["combined_heading", "both", "no_discussion_section"]
    assert articles["heading"].to_list() == [
        "Results AND Discussion", "Discussion & findings", "",
    ]
    assert unreadable == 1
    output = tmp_path / "combined.csv"
    articles.write_csv(output)
    assert pl.read_csv(output, schema_overrides={"pmid": pl.String}).to_dicts() == articles.to_dicts()


def test_unreadable_json_is_tolerated(inputs):
    cleaned, _, _ = inputs
    (cleaned / "5.json").write_text("{invalid", encoding="utf-8")
    articles, unreadable = combined.list_combined_articles(*inputs)
    assert unreadable == 1
    assert articles.filter(pl.col("pmid") == "5")["reason"].item() == "no_discussion_section"


def test_empty_output_summary(inputs, capsys):
    cleaned, corpus_path, inventory_path = inputs
    corpus = pl.read_parquet(corpus_path).with_columns(pl.lit(True).alias("is_non_imrad"))
    corpus.write_parquet(corpus_path)
    articles, unreadable = combined.list_combined_articles(cleaned, corpus_path, inventory_path)
    assert articles.is_empty()
    assert articles.columns == combined.OUTPUT_COLUMNS
    assert unreadable == 0
    combined.print_summary(articles, unreadable)
    summary = capsys.readouterr().out
    assert "reason both: 0" in summary
    assert "journal_field medical_education: 0" in summary
    assert "Total: 0" in summary
