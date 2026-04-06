nmfqp.reg <- function(X, Z, k, H_init = NULL, maxiter = 1000, tol = 1e-6, ncores = 1) {
  runtime <- proc.time()
  res <- nmf_reg_cpp(X, Z, k, H_init, maxiter, tol, ncores)
  runtime <- proc.time() - runtime

  res
}
