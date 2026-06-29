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

GROUPS <- c("must", "should", "need_to", "have_to", "required_to")
ALPHA  <- 0.05
NQ     <- 20            # Gauss-Hermite nodes

# ---- Gauss-Hermite quadrature nodes/weights (Golub-Welsch) ------------------
gauss_hermite <- function(n) {
  i <- 1:(n - 1); b <- sqrt(i / 2)
  J <- matrix(0, n, n); J[cbind(i, i + 1)] <- b; J[cbind(i + 1, i)] <- b
  e <- eigen(J, symmetric = TRUE)
  ord <- order(e$values)
  list(x = e$values[ord], w = (e$vectors[1, ord]^2) * sqrt(pi))
}
GH <- gauss_hermite(NQ)

# ---- Poisson-lognormal marginal negative log-likelihood --------------------
# par = (beta0, beta1, log_tau); X = (1, me); offset = log(words).
make_nll <- function(y, X, offset) {
  lgy <- lgamma(y + 1)
  nodes <- GH$x; logw <- log(GH$w)
  function(par) {
    beta <- par[1:2]; tau <- exp(par[3]); s <- sqrt(2) * tau
    eta <- as.vector(X %*% beta) + offset
    ll_nodes <- matrix(0, length(y), length(nodes))
    for (k in seq_along(nodes)) {
      logmu <- eta + s * nodes[k]
      ll_nodes[, k] <- logw[k] - exp(logmu) + y * logmu - lgy
    }
    mx <- ll_nodes[cbind(seq_len(nrow(ll_nodes)), max.col(ll_nodes))]
    li <- mx + log(rowSums(exp(ll_nodes - mx))) - 0.5 * log(pi)
    -sum(li)
  }
}

fit_pln <- function(count, words, me) {
  X <- cbind(1, me); offset <- log(words)
  nll <- make_nll(count, X, offset)
  start <- c(log(sum(count) / sum(words)), 0, log(0.5))
  opt <- optim(start, nll, method = "BFGS", hessian = TRUE,
               control = list(maxit = 500, reltol = 1e-9))
  if (opt$convergence != 0) {
    opt <- optim(opt$par, nll, method = "Nelder-Mead", hessian = TRUE,
                 control = list(maxit = 2000))
  }
  b0 <- opt$par[1]; b1 <- opt$par[2]; tau <- exp(opt$par[3])
  V <- tryCatch(solve(opt$hessian), error = function(e) matrix(NA, 3, 3))
  se1 <- sqrt(V[2, 2])
  list(rate_ratio = exp(b1),
       ci_lower = exp(b1 - 1.96 * se1), ci_upper = exp(b1 + 1.96 * se1),
       p_value = 2 * pnorm(-abs(b1 / se1)),
       rate_gm = exp(b0 + tau^2 / 2) * 1000,
       rate_me = exp(b0 + b1 + tau^2 / 2) * 1000,
       tau = tau, convergence = opt$convergence)
}

inv  <- as.data.table(read_parquet(inv_path))
corp <- as.data.table(read_parquet(corp_path)[, c("pmid", "journal_field", "is_non_imrad")])

UNITS <- list(
  primary_abstract_body    = c("abstract_pubmed", "introduction", "methods",
                               "results", "discussion", "conclusion"),
  sg_abstract_pubmed       = c("abstract_pubmed"),
  sg_introduction          = c("introduction"),
  sg_methods               = c("methods"),
  sg_results               = c("results"),
  sg_discussion_conclusion = c("discussion", "conclusion")
)

build_unit <- function(secs) {
  sdcols <- intersect(c("word_count", "total_deontic", GROUPS, paste0(GROUPS, "_neg")),
                      names(inv))
  d <- inv[section %in% secs, lapply(.SD, sum), by = pmid, .SDcols = sdcols]
  d <- merge(d, corp, by = "pmid", all.x = TRUE)
  d <- d[is.na(is_non_imrad) | !is_non_imrad][word_count > 0]
  d[, me := as.integer(journal_field == "medical_education")]
  d
}

fit_one <- function(d, count_vec, label) {
  cgm <- sum(count_vec[d$me == 0]); wgm <- sum(d$word_count[d$me == 0])
  cme <- sum(count_vec[d$me == 1]); wme <- sum(d$word_count[d$me == 1])
  r <- fit_pln(count_vec, d$word_count, d$me)
  data.frame(expression_group = label, n_studies = nrow(d),
             count_gm = cgm, count_me = cme, words_gm = wgm, words_me = wme,
             rate_gm = r$rate_gm, rate_me = r$rate_me,
             rate_ratio = r$rate_ratio, ci_lower = r$ci_lower, ci_upper = r$ci_upper,
             p_value = r$p_value, tau = r$tau, convergence = r$convergence)
}

dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

for (uname in names(UNITS)) {
  cat("=== unit:", uname, "===\n"); flush.console()
  d <- build_unit(UNITS[[uname]])
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

  out <- do.call(rbind, rows)
  is_grp <- out$expression_group %in% GROUPS
  out$p_corrected <- NA_real_
  out$p_corrected[is_grp] <- pmin(out$p_value[is_grp] * length(GROUPS), 1)
  out$significant <- NA
  out$significant[is_grp] <- out$p_corrected[is_grp] < ALPHA
  out$significant[out$expression_group == "all"] <-
    out$p_value[out$expression_group == "all"] < ALPHA

  udir <- file.path(out_root, uname)
  dir.create(udir, showWarnings = FALSE, recursive = TRUE)
  write.csv(out, file.path(udir, "meta_results.csv"), row.names = FALSE)
  cat("  -> wrote", file.path(udir, "meta_results.csv"), "\n\n"); flush.console()
}

cat("ALL META-ANALYSES COMPLETE\n")
