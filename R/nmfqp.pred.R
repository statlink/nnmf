nmfqp.pred <- function(xnew, H, ridge = 1e-8, ncores = 1) {

  runtime <- proc.time()
  res <- nmf_pred_cpp(xnew, H, ridge, ncores)
  runtime <- proc.time() - runtime

  list(Wnew = res$Wnew, Znew = res$Znew, runtime = runtime)
}
