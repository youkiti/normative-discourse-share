# Shared Poisson-lognormal model; see meta_analysis_deontic.R for model details.

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

UNITS <- list(
  primary_abstract_body    = c("abstract_pubmed", "introduction", "methods",
                               "results", "discussion", "conclusion"),
  sg_abstract_pubmed       = c("abstract_pubmed"),
  sg_introduction          = c("introduction"),
  sg_methods               = c("methods"),
  sg_results               = c("results"),
  sg_discussion_conclusion = c("discussion", "conclusion")
)

build_unit <- function(inv, corp, secs) {
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

build_results <- function(rows) {
  out <- do.call(rbind, rows)
  is_grp <- out$expression_group %in% GROUPS
  out$p_corrected <- NA_real_
  out$p_corrected[is_grp] <- pmin(out$p_value[is_grp] * length(GROUPS), 1)
  out$significant <- NA
  out$significant[is_grp] <- out$p_corrected[is_grp] < ALPHA
  out$significant[out$expression_group == "all"] <-
    out$p_value[out$expression_group == "all"] < ALPHA

  out
}
