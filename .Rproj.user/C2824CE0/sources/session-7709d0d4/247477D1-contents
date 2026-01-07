snmfskl.eg <- function(x, k, W = NULL, H = NULL, lr_w = 0.1, lr_h = 0.1, maxiter = 1000, tol = 1e-6, clip_exp = 50) {
  
  runtime <- proc.time()
  n <- dim(x)[1]  ;  D <- dim(x)[2]
  # Initialize W, H randomly on simplex
  if ( is.null(W) ) {
    W <- matrix( Rfast2::Runif(n * k), nrow = n, ncol = k)
    W <- W / Rfast::rowsums(W)
  }	
  if ( is.null(H) ) {
    H <- matrix( Rfast2::Runif(k * D), nrow = k, ncol = D)
    H <- H / Rfast::rowsums(H)
  }
  
  prev_obj <- Inf
  mat <- matrix(1, n, D)
  lx <- log(x)

  for ( iter in 1:maxiter ) {
    Z <- W %*% H              # model reconstruction
    R <- x / Z                # element-wise division
    # ---- Gradient for symmetrised KL divergence ----  
    log_diff <- log(Z) - lx
    grad_common <- mat - R + log_diff
    # Gradient w.r.t. W: (1 - R + log(Z) - log(X)) H^T
    grad_w <- tcrossprod(grad_common, H)
    # Gradient w.r.t. H: W^T (1 - R + log(Z) - log(X))
    grad_h <- crossprod(W, grad_common)
    # ---- Exponentiated-gradient updates ----
    expo_w <- -lr_w * grad_w
    expo_w <- pmax( pmin(expo_w, clip_exp), -clip_exp )
    W <- W * exp(expo_w)
    W <- W / Rfast::rowsums(W)
    expo_h <- -lr_h * grad_h
    expo_h <- pmax(pmin(expo_h, clip_exp), -clip_exp)
    H <- H * exp(expo_h)
    H <- H / Rfast::rowsums(H)
    # ---- Compute symmetrised KL divergence ----
    Z <- W %*% H
    obj <- sum( (x - Z) * ( lx - log(Z) ), na.rm = TRUE )
    if ( abs(prev_obj - obj ) < tol) {
      break
    }
    prev_obj <- obj
  }
  
  runtime <- proc.time() - runtime
  
  list(W = W, H = H, Z = Z, obj = obj, iters = iter, runtime = runtime)
}