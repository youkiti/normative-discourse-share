"""CLI entry point for normative-discourse (reproduction subset).

Usage:
    python -m normative_discourse.cli.main [COMMAND] [OPTIONS]

This open-reproduction build exposes only the three commands needed to regenerate
the manuscript tables from the shared text corpus:

    build-corpus       cleaned-text JSON -> analysis-corpus Parquet
    detect-deontic     deontic-marker detection per section
    analyze-frequency  G-squared frequency analysis (secondary)

The upstream retrieval / PDF-extraction commands are intentionally omitted; the
reproduction path never goes online.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from normative_discourse import __version__
from normative_discourse.config.settings import get_settings, setup_logging

app = typer.Typer(
    name="normative-discourse",
    help="Deontic expression analysis in medical literature (reproduction subset).",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
) -> None:
    """normative-discourse CLI."""
    if version:
        console.print(f"normative-discourse v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def build_corpus(
    output: Optional[Path] = typer.Option(
        None,
        help="Output Parquet path (default: data/processed/articles_corpus.parquet)",
    ),
    articles_csv: Optional[Path] = typer.Option(
        None,
        help="Articles CSV (default: data/raw/pubmed/articles_2025_included.csv)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print statistics only, do not write Parquet",
    ),
    fetch_pubmed_abstracts: bool = typer.Option(
        True,
        "--fetch-pubmed-abstracts/--no-fetch-pubmed-abstracts",
        help="Fetch abstracts from PubMed API (cached in data/interim/pubmed_abstracts.json)",
    ),
    apply_screening: bool = typer.Option(
        True,
        "--apply-screening/--no-apply-screening",
        help="Apply AI screening + human review filter (screening_results.csv + review TSVs)",
    ),
    overrides: Optional[Path] = typer.Option(
        None,
        "--overrides",
        help="Section overrides TOML (default: config/section_overrides.toml if exists)",
    ),
    no_overrides: bool = typer.Option(
        False,
        "--no-overrides",
        help="Skip manual section overrides even if TOML exists",
    ),
) -> None:
    """Build analysis corpus Parquet from metadata + cleaned-text JSONs."""
    import json as _json

    from normative_discourse.config.settings import PROJECT_ROOT, get_paths_config
    from normative_discourse.data_io.corpus_builder import build_corpus as _build
    from normative_discourse.data_io.corpus_builder import save_corpus

    settings = get_settings()
    setup_logging(settings.log_level)

    paths_config = get_paths_config()

    if articles_csv is None:
        articles_csv = (
            PROJECT_ROOT
            / paths_config.get("raw_pubmed", "data/raw/pubmed")
            / "articles_2025_included.csv"
        )
    cleaned_text_dir = PROJECT_ROOT / "data" / "interim" / "cleaned_text"
    med_teach_csv = PROJECT_ROOT / "data" / "processed" / "med_teach_article_types.csv"
    abstracts_cache = PROJECT_ROOT / "data" / "interim" / "pubmed_abstracts.json"

    if output is None:
        output = PROJECT_ROOT / "data" / "processed" / "articles_corpus.parquet"

    if not articles_csv.exists():
        console.print(f"[red]Articles CSV not found: {articles_csv}[/red]")
        raise typer.Exit(code=1)

    console.print("[bold]Build analysis corpus[/bold]")
    console.print(f"  Articles CSV: [dim]{articles_csv}[/dim]")
    console.print(f"  Cleaned text: [dim]{cleaned_text_dir}[/dim]")

    # Fetch PubMed abstracts
    pubmed_abstracts: dict[str, str] | None = None
    if fetch_pubmed_abstracts:
        if abstracts_cache.exists():
            console.print(f"  Loading cached PubMed abstracts: [dim]{abstracts_cache}[/dim]")
            with open(abstracts_cache, encoding="utf-8") as f:
                pubmed_abstracts = _json.load(f)
            console.print(f"  Cached abstracts: [green]{len(pubmed_abstracts)}[/green]")
        else:
            from normative_discourse.data_io.pubmed_client import (
                efetch_by_pmids,
                extract_abstracts,
            )

            settings.validate_required("pubmed")
            # Read PMIDs from CSV (all included rows, before Stage 2 filter)
            import pandas as _pd

            meta = _pd.read_csv(articles_csv, dtype=str)
            meta = meta[meta["included"].str.lower() == "true"]
            pmids = meta["pmid"].dropna().tolist()
            console.print(f"  Fetching abstracts for [bold]{len(pmids)}[/bold] PMIDs from PubMed...")

            xml_text = efetch_by_pmids(
                pmids=pmids,
                email=settings.pubmed_email,
                api_key=settings.pubmed_api_key,
            )
            pubmed_abstracts = extract_abstracts(xml_text)
            console.print(
                f"  Fetched [green]{len(pubmed_abstracts)}[/green] / {len(pmids)} abstracts"
            )

            # Cache
            abstracts_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(abstracts_cache, "w", encoding="utf-8") as f:
                _json.dump(pubmed_abstracts, f, ensure_ascii=False)
            console.print(f"  Cached: [dim]{abstracts_cache}[/dim]")

    # Resolve overrides TOML path
    overrides_toml: Path | None = None
    if not no_overrides:
        if overrides is not None:
            overrides_toml = overrides
        else:
            default_ov = PROJECT_ROOT / "config" / "section_overrides.toml"
            if default_ov.exists():
                overrides_toml = default_ov
    if overrides_toml:
        console.print(f"  Section overrides: [cyan]{overrides_toml}[/cyan]")

    df = _build(
        articles_csv=articles_csv,
        cleaned_text_dir=cleaned_text_dir,
        med_teach_csv=med_teach_csv if med_teach_csv.exists() else None,
        pubmed_abstracts=pubmed_abstracts,
        overrides_toml=overrides_toml,
    )

    # Apply screening filter (AI + human review)
    if apply_screening:
        from normative_discourse.data_io.corpus_builder import apply_screening_filter

        processed_dir = PROJECT_ROOT / "data" / "processed"
        screening_csv = processed_dir / "screening_results.csv"
        if screening_csv.exists():
            review_tsvs = sorted(processed_dir.glob("screening_*review*.tsv"))
            n_before = len(df)
            df = apply_screening_filter(df, screening_csv, review_tsvs)
            console.print(
                f"  Screening filter: [bold]{n_before}[/bold] → [green]{len(df)}[/green]"
                f" ({n_before - len(df)} excluded)"
            )
        else:
            console.print("  [yellow]screening_results.csv not found — skipping screening filter[/yellow]")

    # Summary table
    summary_table = Table(title="Corpus Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="bold")
    summary_table.add_row("Total articles", str(len(df)))
    summary_table.add_row(
        "With body text",
        str(df["has_body"].sum()),
    )
    summary_table.add_row(
        "Without body text",
        str((~df["has_body"]).sum()),
    )
    console.print()
    console.print(summary_table)

    # By-journal breakdown
    journal_counts = (
        df.groupby("journal", observed=True)
        .agg(
            total=("pmid", "count"),
            with_body=("has_body", "sum"),
            avg_words=("word_count", "mean"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )
    console.print()
    jt = Table(title="By Journal")
    jt.add_column("Journal", style="cyan")
    jt.add_column("Total", justify="right")
    jt.add_column("With body", justify="right", style="green")
    jt.add_column("Avg words", justify="right", style="dim")
    for _, r in journal_counts.iterrows():
        jt.add_row(
            str(r["journal"]),
            str(r["total"]),
            str(int(r["with_body"])),
            str(int(r["avg_words"])),
        )
    console.print(jt)

    if dry_run:
        console.print()
        console.print("[yellow]Dry run -- no file written.[/yellow]")
    else:
        save_corpus(df, output)
        console.print()
        size_mb = output.stat().st_size / (1024 * 1024)
        console.print(
            f"  Saved: [dim]{output}[/dim] ({size_mb:.1f} MB, {len(df)} rows)"
        )


@app.command()
def detect_deontic(
    input_parquet: Optional[Path] = typer.Option(
        None,
        help="Input corpus Parquet (default: data/processed/articles_corpus.parquet)",
    ),
    inventory_output: Optional[Path] = typer.Option(
        None,
        help="Output section_inventory.parquet (default: data/processed/section_inventory.parquet)",
    ),
    matches_output: Optional[Path] = typer.Option(
        None,
        help="Output matches.parquet (default: data/processed/matches.parquet)",
    ),
    sections: Optional[str] = typer.Option(
        None,
        help="Comma-separated section names to analyse (default: abstract_pubmed)",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip articles already in output (default: on)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run on --limit articles and print results without saving",
    ),
    limit: Optional[int] = typer.Option(
        None,
        help="Process at most N articles (default: all). Implied 10 for --dry-run.",
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
    log_file: Optional[Path] = typer.Option(None, help="Log file path"),
) -> None:
    """Run deontic expression detection on corpus articles."""
    from normative_discourse.config.settings import get_detection_config, get_paths_config
    from normative_discourse.detection.deontic_detector import (
        DeonticDetector,
        run_detection_on_corpus,
    )

    setup_logging(log_level, log_file)
    paths_cfg = get_paths_config()

    corpus_path = input_parquet or Path("data/processed/articles_corpus_10yr_clean.parquet")
    inv_path = inventory_output or Path(paths_cfg.get("detection_inventory", "data/processed/section_inventory.parquet"))
    match_path = matches_output or Path(paths_cfg.get("detection_matches", "data/processed/matches.parquet"))

    section_list = sections.split(",") if sections else None

    if dry_run:
        # Dry-run: detect on limited articles and print results.
        effective_limit = limit or 10
        import polars as pl

        df = pl.read_parquet(corpus_path)
        det_cfg = get_detection_config()
        sec_list = section_list or det_cfg.get("default_sections", ["abstract_pubmed"])

        detector = DeonticDetector()

        pmids = df["pmid"].to_list()[:effective_limit]
        total_matches = 0

        for pmid in pmids:
            row_data = df.filter(pl.col("pmid") == pmid).to_dicts()[0]
            journal = row_data.get("journal", "")
            title = row_data.get("title", "")

            for sec in sec_list:
                raw_text = row_data.get(sec) or ""
                inv, matches = detector.analyze_section(pmid, sec, raw_text)

                header = f"[bold cyan]{pmid}[/bold cyan] ({journal}) — {sec}: {inv.word_count} words, {inv.total_deontic} deontic"
                console.print(header)

                if matches:
                    match_table = Table(show_header=True, box=None, pad_edge=False)
                    match_table.add_column("Expression", style="green")
                    match_table.add_column("Group")
                    match_table.add_column("Neg?")
                    match_table.add_column("Context", style="dim")
                    for m in matches:
                        neg_str = "[red]NEG[/red]" if m.is_negation else ""
                        ctx = f"...{m.context_before} [{m.expression}] {m.context_after}..."
                        match_table.add_row(m.expression, m.expression_group, neg_str, ctx)
                    console.print(match_table)
                console.print()
                total_matches += len(matches)

        console.print(f"[bold]Dry-run complete: {len(pmids)} articles, {total_matches} matches[/bold]")
        return

    # Full run.
    if not corpus_path.exists():
        console.print(f"[red]Corpus not found: {corpus_path}[/red]")
        raise typer.Exit(1)

    n_articles, n_matches = run_detection_on_corpus(
        parquet_path=corpus_path,
        inventory_path=inv_path,
        matches_path=match_path,
        sections=section_list,
        resume=resume,
        limit=limit,
    )
    console.print(
        f"[green]Detection complete:[/green] {n_articles} articles, {n_matches} matches"
    )
    console.print(f"  Inventory: [dim]{inv_path}[/dim]")
    console.print(f"  Matches:   [dim]{match_path}[/dim]")


@app.command()
def analyze_frequency(
    inventory_parquet: Optional[Path] = typer.Option(
        None,
        help="Input section_inventory.parquet",
    ),
    corpus_parquet: Optional[Path] = typer.Option(
        None,
        help="Corpus Parquet for metadata (journal, field)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for tables and report JSON",
    ),
    section: str = typer.Option(
        "abstract_pubmed",
        help="Analysis label / single section to analyse (must match detection data)",
    ),
    sections: Optional[str] = typer.Option(
        None,
        help="Comma-separated sections to combine into one analysis unit "
        "(e.g. 'abstract_pubmed,introduction,methods,results,discussion,conclusion'). "
        "When set, per-article tallies are summed across these sections.",
    ),
    imrad_only: bool = typer.Option(
        False,
        "--imrad-only/--no-imrad-only",
        help="Exclude non-IMRaD records so the denominator is the IMRaD subset only",
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
    log_file: Optional[Path] = typer.Option(None, help="Log file path"),
) -> None:
    """Compute frequency statistics and run G-squared tests."""
    from normative_discourse.config.settings import get_paths_config
    from normative_discourse.stats.frequency_analysis import run_frequency_analysis

    setup_logging(log_level, log_file)
    paths_cfg = get_paths_config()

    inv_path = inventory_parquet or Path(paths_cfg.get("detection_inventory", "data/processed/section_inventory.parquet"))
    corp_path = corpus_parquet or Path("data/processed/articles_corpus_10yr_clean.parquet")
    out_dir = output_dir or Path(paths_cfg.get("results_output", "results/output")) / section

    section_list = [s.strip() for s in sections.split(",")] if sections else None

    if not inv_path.exists():
        console.print(f"[red]Inventory not found: {inv_path}[/red]")
        console.print("Run [bold]detect-deontic[/bold] first.")
        raise typer.Exit(1)

    report = run_frequency_analysis(
        inventory_path=inv_path,
        corpus_path=corp_path,
        output_dir=out_dir,
        section=section,
        sections=section_list,
        imrad_only=imrad_only,
    )

    console.print(f"[green]Frequency analysis complete[/green] - section: {section}")
    console.print(f"  Articles: {report.total_articles}")
    console.print(f"  Total words: {report.total_words:,}")
    console.print(f"  Total deontic: {report.total_deontic}")
    console.print(f"  Output: [dim]{out_dir}[/dim]")

    # Print G-test summary table.
    if report.g_test_results:
        g_table = Table(title="G-squared Test Results (GM vs ME)")
        g_table.add_column("Group", style="cyan")
        g_table.add_column("Rate GM", justify="right")
        g_table.add_column("Rate ME", justify="right")
        g_table.add_column("Rate ratio [95% CI]", justify="right")
        g_table.add_column("G-sq", justify="right")
        g_table.add_column("p (corr)", justify="right")
        g_table.add_column("Sig?", justify="center")
        for r in report.g_test_results:
            sig = "[green]Yes[/green]" if r.significant else "No"
            rr_ci = f"{r.rate_ratio:.2f} [{r.rate_ratio_ci_lower:.2f}, {r.rate_ratio_ci_upper:.2f}]"
            g_table.add_row(
                r.expression_group,
                f"{r.rate_group1:.3f}",
                f"{r.rate_group2:.3f}",
                rr_ci,
                f"{r.g_squared:.2f}",
                f"{r.p_value_corrected:.2e}",
                sig,
            )
        console.print(g_table)


if __name__ == "__main__":
    app()
