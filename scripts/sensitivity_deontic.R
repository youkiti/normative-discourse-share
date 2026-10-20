#!/usr/bin/env Rscript
# Sensitivity analyses for the article-level Poisson-lognormal model.
# Usage: Rscript scripts/sensitivity_deontic.R [inv] [corp] [combined_csv] [out_root]

suppressMessages({
  library(arrow)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
inv_path <- if (length(args) >= 1) args[1] else "data/processed/section_inventory.parquet"
corp_path <- if (length(args) >= 2) args[2] else "data/processed/articles_corpus_10yr_clean.parquet"
combined_path <- if (length(args) >= 3) args[3] else "results/output_sensitivity/combined_results_articles.csv"
out_root <- if (length(args) >= 4) args[4] else "results/output_sensitivity"

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
lib_path <- if (length(script_arg) > 0) {
  file.path(dirname(sub("^--file=", "", script_arg[1])), "meta_pln_lib.R")
} else {
  "scripts/meta_pln_lib.R"
}
source(lib_path)

inv <- as.data.table(read_parquet(inv_path))
corp <- as.data.table(read_parquet(corp_path)[, c("pmid", "journal_field", "is_non_imrad", "journal")])
combined <- read.csv(combined_path, colClasses = c(pmid = "character"))
units <- c("primary_abstract_body", "sg_results")
journals <- sort(unique(corp$journal))
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

report_fit <- function(d, count_vec, label) {
  r <- fit_one(d, count_vec, label)
  cat(sprintf("  %-12s RR=%.3f (%.3f-%.3f) p=%.2e tau=%.3f rGM=%.3f rME=%.3f conv=%d\n",
              label, r$rate_ratio, r$ci_lower, r$ci_upper, r$p_value, r$tau,
              r$rate_gm, r$rate_me, r$convergence))
  flush.console()
  r
}

lojo_row <- function(uname, journal, field, d, r) {
  data.frame(unit = uname, omitted_journal = journal, omitted_field = field,
             n_studies = r$n_studies, n_gm = sum(d$me == 0), n_me = sum(d$me == 1),
             rate_gm = r$rate_gm, rate_me = r$rate_me,
             rate_ratio = r$rate_ratio, ci_lower = r$ci_lower, ci_upper = r$ci_upper,
             p_value = r$p_value, tau = r$tau, convergence = r$convergence)
}

format_rr <- function(r) {
  sprintf("%.2f [%.2f, %.2f]", r$rate_ratio, r$ci_lower, r$ci_upper)
}

lojo_rows <- list()
exclusion_lines <- character()
for (uname in units) {
  cat("=== unit:", uname, "===\n")
  d <- build_unit(inv, corp, UNITS[[uname]])
  excluded <- d[!pmid %in% combined$pmid]
  cat(sprintf("  n before=%d / after=%d\n", nrow(d), nrow(excluded)))
  flush.console()

  cat("  Full data\n")
  full <- report_fit(d, d$total_deontic, "all")
  lojo_rows[[length(lojo_rows) + 1]] <- lojo_row(uname, "(none)", NA_character_, d, full)

  cat("  Excluding combined Results articles\n")
  rows <- list()
  for (g in GROUPS) {
    cnt <- excluded[[g]] + excluded[[paste0(g, "_neg")]]
    rows[[g]] <- report_fit(excluded, cnt, g)
  }
  rows[["all"]] <- report_fit(excluded, excluded$total_deontic, "all")
  out <- build_results(rows)
  udir <- file.path(out_root, "excl_combined", uname)
  dir.create(udir, showWarnings = FALSE, recursive = TRUE)
  write.csv(out, file.path(udir, "meta_results.csv"), row.names = FALSE)
  exclusion_lines <- c(exclusion_lines, sprintf("| %s | %d | %d | %s | %s |",
                       uname, nrow(d), nrow(excluded), format_rr(full),
                       format_rr(rows[["all"]])))

  for (journal_name in journals) {
    cat("  Omitting journal:", journal_name, "\n")
    flush.console()
    omitted <- d[journal != journal_name]
    field <- paste(sort(unique(corp[journal == journal_name, journal_field])), collapse = ", ")
    r <- report_fit(omitted, omitted$total_deontic, "all")
    lojo_rows[[length(lojo_rows) + 1]] <- lojo_row(uname, journal_name, field, omitted, r)
    write.csv(do.call(rbind, lojo_rows), file.path(out_root, "lojo_all.csv"), row.names = FALSE)
  }
}
lojo <- do.call(rbind, lojo_rows)
write.csv(lojo, file.path(out_root, "lojo_all.csv"), row.names = FALSE)

input_paths <- c(inv_path, corp_path, combined_path)
input_lines <- vapply(input_paths, function(path) {
  sprintf("- Input: %s; modification time: %s",
          normalizePath(path, winslash = "/", mustWork = TRUE),
          format(file.info(path)$mtime, "%Y-%m-%d %H:%M:%S %Z"))
}, character(1))
lojo_lines <- vapply(seq_len(nrow(lojo)), function(i) {
  r <- lojo[i, ]
  sprintf("| %s | %s | %d | %s |", r$unit, r$omitted_journal, r$n_studies, format_rr(r))
}, character(1))
range_lines <- vapply(units, function(uname) {
  rr <- lojo$rate_ratio[lojo$unit == uname & lojo$omitted_journal != "(none)"]
  sprintf("- %s: LOJO RR min %.2f; max %.2f.", uname, min(rr), max(rr))
}, character(1))
summary_lines <- c(
  "# Deontic-marker sensitivity analyses",
  "",
  "The corpus commit is not known to this script. Input paths and modification times:",
  "",
  input_lines,
  "",
  "## Combined-section exclusion",
  "",
  "| Unit | N before | N after | Full RR [95% CI] | Excluded RR [95% CI] |",
  "| --- | ---: | ---: | --- | --- |",
  exclusion_lines,
  "",
  "## Leave-one-journal-out (all markers)",
  "",
  "| Unit | Omitted journal | N | RR [95% CI] |",
  "| --- | --- | ---: | --- |",
  lojo_lines,
  "",
  range_lines
)
writeLines(summary_lines, file.path(out_root, "summary.md"))
cat("ALL SENSITIVITY ANALYSES COMPLETE\n")
flush.console()
