nmf.hals <- function(x, k, maxiter = 2000, tol = 1e-6, history = FALSE) {

  runtime <- proc.time()
  n <- dim(x)[1]  ;  D <- dim(x)[2]

  W <- matrix( rangen::Runif(n * k), nrow = n, ncol = k )
  H <- matrix( rangen::Runif(k * D), nrow = k, ncol = D )
  R <- x - W %*% H  # Maintain residual
  obj <- numeric(maxiter)

  for ( iter in 1:maxiter ) {
    # Update W
    for ( j in 1:k ) {
      # Add back current component
      R <- R + tcrossprod(W[, j], H[j, ])
      # Update
      norm_sq <- sum(H[j, ]^2)
      if ( norm_sq > 1e-10 ) {
        W[, j] <- pmax( 0, (R %*% H[j, ]) / norm_sq )
      }
      # Subtract new component
      R <- R - tcrossprod(W[, j], H[j, ])
    }
    # Update H
    for ( j in 1:k ) {
      R <- R + tcrossprod(W[, j], H[j, ])
      norm_sq <- sum(W[, j]^2)
      if ( norm_sq > 1e-10 ) {
        H[j, ] <- pmax( 0, crossprod(W[, j], R) / norm_sq )
      }
      R <- R - tcrossprod(W[, j], H[j, ])
    }
    obj[iter] <- sum(R^2)

    if ( iter > 1 && abs( obj[iter - 1] - obj[iter] ) < tol ) {
      obj <- obj[1:iter]
      break
    }
  }  ##  end  for ( iter in 1:maxiter )

  error <- obj
  if ( !history )  error <- NULL
  obj <- obj[iter]
  colnames(H) <- colnames(x)
  Z <- W %*% H
  runtime <- proc.time() - runtime

  list(W = W, H = H, Z = Z, obj = obj, error = error, iters = iter, runtime = runtime )
}
