# Results marker speaker check

Classify deontic markers in Results into five categories, in this order:
`participant`, `participant_summary`, `author`, `other_source`, `unclear`.
`participant` reproduces participants' own words; `participant_summary` reports
what participants said, felt, wanted or recommended in the authors' own words. Run from the repository root with
`PYTHONPATH=src` (PowerShell: `$env:PYTHONUTF8=1; $env:PYTHONPATH='src'`).
Inputs and generated review data live in `data/processed/quote_check/`, which is
gitignored. The default report is `results/quote_check/summary.md` and contains
ids and counts, with no passage text.

## Public labels

Run `python scripts/quote_check/tally_speaker.py --labels data/quote_check/speaker_labels.csv`
to reproduce the tally without corpus access. With no `--sample`, an existing
default labels file is selected automatically. `--labels-meta` defaults to the
sibling `speaker_labels.meta.json`. The report records both sample and labels
SHA-256 hashes, including the raw sample hash for provenance and its LF-normalized
hash for cross-platform checks; evidence validation is reported from the upstream metadata.
The CSV has exactly these columns, in frozen sample order:

```text
id,pmid,journal,journal_field,expression,expression_group,start_char,end_char,sonnet_v1_speaker,sonnet_speaker,human_speaker,human_auto
```

`human_auto` is `true` when the judged human document's note starts with `auto:`,
and `false` otherwise. A blank human category is unjudged. The first-round column
may be entirely empty, which omits its section from the report; partial coverage
is rejected. Categories, fields, unique ids, offsets and the exact header are validated.
`python reproducibility/run_08_speaker_check.py` also redraws and checks the sample
when the corpus, matches and combined-section CSV are available. The sample SHA-256
check uses LF-normalized bytes so it holds on any OS. The redrawn sample stays in gitignored `reproducibility/work/quote_check/`.

## Private rating workflow

1. Freeze the sample: `python scripts/quote_check/sample_results_markers.py`.
   Defaults to 100 markers per field, seed 20261018, excluding non-IMRaD and
   combined-section articles. Existing samples cannot be overwritten.
2. Prepare rater files: `python scripts/quote_check/make_batches.py`.
3. The draft ratings were made by Claude Sonnet subagents in Claude Code,
   each reading one `sonnet/prompt_NN.md` derived from `prompt_template.md`
   and writing its corresponding `sonnet/out_NN.json`. Each prompt restricts the
   rater to its own input and output files. This is a separate rating step;
   these scripts do not call a model or start subagents.
4. Build the page: `python scripts/quote_check/build_review_page.py`.
5. Run `python scripts/quote_check/auto_quoted.py` to flag markers enclosed in
   quotation marks or digit-free square brackets, scanning the normalized Results
   text from its start (the normalizer removes line breaks, so this is the whole
   section; `--sample`, `--corpus`, and `--out` override the defaults).
   For flagged records not yet judged, write `speaker: "participant"` with a
   `note: "auto: ..."` at the same level into the review page's database; this is
   a mechanical classification, and the script only produces `auto_quoted.json`.
6. Review `speaker_review.html` as a claude.ai artifact with its database enabled.
   Turn on "未判定のみ" (unjudged only) to skip judged records during navigation
   and advance to the next unjudged record after saving; the filter is remembered
   in local storage.
   Export the database using the Artifact data tool, retaining one JSON file per
   document in `data/processed/quote_check/human/`.
7. Tally with explicit private inputs:
   `python scripts/quote_check/tally_speaker.py --sample data/processed/quote_check/sample.json --human data/processed/quote_check/human`.
   The Markdown report is also printed to stdout. `--model-rr` defaults to 16.94.
8. Export public labels from the validated private inputs:

   ```bash
   python scripts/quote_check/export_labels.py \
     --sample data/processed/quote_check/sample.json \
     --sonnet-dir data/processed/quote_check/sonnet \
     --sonnet-v1-dir data/processed/quote_check/sonnet_v1 \
     --human data/processed/quote_check/human
   ```

   The outputs are `data/quote_check/speaker_labels.csv` and its sibling
   `speaker_labels.meta.json` (raw and LF-normalized sample hashes, fixed seed, per-field sample size,
   context window and ids with unverifiable evidence). `--out` changes the CSV
   path. Existing CSV or metadata files are refused unless `--overwrite` is given.
   The exporter reuses the tally's Sonnet and human loaders and writes only the
   public columns above. The frozen configuration is seed 20261018, 100 per field,
   window 600. Do not export a differently configured sample as the study labels.

The sampler accepts `--corpus`, `--matches`, `--combined`, `--out`, and
`--n-per-field`. Subsequent steps accept `--sample`; batch generation accepts
`--out-dir`, page building and tally accept `--sonnet-dir`, page building accepts
`--out`, and tally accepts `--report` and `--sonnet-v1-dir`.
The first Sonnet round used four categories and is kept in `sonnet_v1/`.
Tally defaults `--sonnet-v1-dir` to `data/processed/quote_check/sonnet_v1`;
when present, it validates that round with the old four-category list and reports
per-field counts and exact CIs separately. An absent directory is silently skipped. Batches contain 20 records, with a shorter
final batch if needed. Tally validates output order against sample chunks and,
when present, batch inputs, checks categories and complete Sonnet coverage, and
lists ids whose evidence is empty or not verbatim.

Human export files may store `speaker` at the top level or under `data`.
Documents are keyed by their `id` when it matches a sample id, otherwise by their
filename stem. Missing/blank speakers are unjudged; duplicate or unknown judged
documents and invalid categories are rejected. Human counts use judged records
only. Each field starts with a grouped human table: participant voice
(`participant` + `participant_summary`), author or document (`author` +
`other_source`), and `unclear`, with exact 95% CIs. Overall and per-field counts
also show judged human documents whose note starts with `auto:` and how many of
these Sonnet classified as `participant`. Old human files containing the original four values remain valid without
mapping. Cohen's kappa uses all five categories. Agreement is not independent
because the human sees Sonnet's draft.

The context window is 600 characters on each side of the marker, so attributions
outside it can be missed. Human participant shares are reported per field for
quotations alone and for quotations plus participant summaries, each with an exact
Clopper-Pearson 95% CI. Crude scaling reports both removals using
`model RR × (1 − p_removed,ME) / (1 − p_removed,GM)`; each is a crude
approximation, not a re-estimated rate ratio.

## Copyright

Article passages, Sonnet's verbatim evidence and reasons, notes, and the generated
review HTML are private and are not distributed. Keep them under gitignored
`data/processed/quote_check/`. Only identifiers, journal, field, grammatical marker
expressions, character offsets, category labels and the mechanical-classification
flag are exported. No context columns are written. The public labels directory
is intentionally not ignored. Anyone with corpus access can regenerate the
passages using the sampler. The human made the final judgements after seeing
Sonnet's drafts; agreement is not independent. Thirty records were mechanically
classified `participant` based on quotation marks or digit-free square brackets.
