#!/usr/bin/env Rscript
# =============================================================================
# Random-effects meta-analysis of per-article deontic-marker rates (ME vs GM).
#
# Unit of analysis = the individual article (each article is one "study").
# Each article contributes a marker count and a word count. The rate is modelled
# with a Poisson-lognormal random-effects model -- the generalized-linear-mixed-
# model realisation of a random-effects meta-analysis of incidence rates
# (equivalent to metafor::rma.glmm, measure = "IRLN", with field as a
# moderator):
#
#     count_i | u_i ~ Poisson( words_i * exp(beta0 + beta1*me_i + u_i) )
#     u_i ~ N(0, tau^2)
#
# An article-level random intercept u_i absorbs between-article heterogeneity
# (tau^2). exp(beta1) is the incidence-rate ratio (ME / GM) with a 95% CI that
# accounts for that heterogeneity -- NOT the corpus-pooled "bag of words" ratio
# used previously, which weighted articles by length and assumed every word was
# an independent Bernoulli trial (yielding spuriously narrow intervals).
#
# Why this estimator and not an inverse-variance log-rate meta-analysis: many
# articles contain zero markers in a given section (up to ~90%), and the zero
# fraction differs between fields; a log transform with a continuity correction
# is then dominated by the correction and produces artefactual estimates (we
# verified sign reversals in the Introduction subgroup). A Poisson likelihood
# handles zeros natively, so no continuity correction is needed.
#
# The marginal likelihood is integrated over u_i by 20-point Gauss-Hermite
# quadrature and maximised by BFGS; standard errors come from the observed
# information (numerical Hessian). Implementation is pure base R -- metafor/meta
# compiled routines segfault on this ARM64 Windows R build.
#
# Output (per analysis unit, under results/output_meta/<unit>/):
#   meta_results.csv -- one row per modal group + an "all" row, with the
#   model-based per-1,000-word rate per field, the rate ratio + 95% CI, the
#   Wald p-value (and Bonferroni-corrected p across the five modal groups),
#   and the between-article heterogeneity SD (tau).
# =============================================================================

suppressMessages({
  library(arrow)        # parquet IO only
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
inv_path  <- if (length(args) >= 1) args[1] else "data/processed/section_inventory.parquet"
corp_path <- if (length(args) >= 2) args[2] else "data/processed/articles_corpus_10yr_clean.parquet"
out_root  <- if (length(args) >= 3) args[3] else "results/output_meta"

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
lib_path <- if (length(script_arg) > 0) {
  file.path(dirname(sub("^--file=", "", script_arg[1])), "meta_pln_lib.R")
} else {
  "scripts/meta_pln_lib.R"
}
source(lib_path)

inv  <- as.data.table(read_parquet(inv_path))
corp <- as.data.table(read_parquet(corp_path)[, c("pmid", "journal_field", "is_non_imrad")])

dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

for (uname in names(UNITS)) {
  cat("=== unit:", uname, "===\n"); flush.console()
  d <- build_unit(inv, corp, UNITS[[uname]])
  cat(sprintf("  n=%d (GM %d / ME %d)\n", nrow(d), sum(d$me == 0), sum(d$me == 1)))
  flush.console()

  rows <- list()
  for (g in GROUPS) {
    cnt <- d[[g]] + d[[paste0(g, "_neg")]]
    rows[[g]] <- fit_one(d, cnt, g)
    r <- rows[[g]]
    cat(sprintf("  %-12s RR=%.3f (%.3f-%.3f) p=%.2e tau=%.3f rGM=%.3f rME=%.3f conv=%d\n",
                g, r$rate_ratio, r$ci_lower, r$ci_upper, r$p_value, r$tau,
                r$rate_gm, r$rate_me, r$convergence)); flush.console()
  }
  rows[["all"]] <- fit_one(d, d$total_deontic, "all")
  r <- rows[["all"]]
  cat(sprintf("  %-12s RR=%.3f (%.3f-%.3f) p=%.2e tau=%.3f rGM=%.3f rME=%.3f conv=%d\n",
              "all", r$rate_ratio, r$ci_lower, r$ci_upper, r$p_value, r$tau,
              r$rate_gm, r$rate_me, r$convergence)); flush.console()

  out <- build_results(rows)

  udir <- file.path(out_root, uname)
  dir.create(udir, showWarnings = FALSE, recursive = TRUE)
  write.csv(out, file.path(udir, "meta_results.csv"), row.names = FALSE)
  cat("  -> wrote", file.path(udir, "meta_results.csv"), "\n\n"); flush.console()
}

cat("ALL META-ANALYSES COMPLETE\n")
