nmf.qp <- function(X, k, H_init = NULL, k_means = TRUE, bs = 1, lr_h = 0.1, tol = 1e-6, maxiter = 1000, ridge = 1e-8, ncores = 1) {
  runtime <- proc.time()

  if (k_means && is.null(H_init)) {
    H <- nnmf::init(X, k, bs)
  }

  res <- nmf_als_cpp(X, k, H_init, lr_h, tol, maxiter, ridge, ncores)

  runtime <- proc.time() - runtime
  colnames(H) <- colnames(X)

  list(W = res$W, H = res$H, Z = res$Z, obj = res$obj, iters = res$iter, runtime = runtime)
}
