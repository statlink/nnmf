nmfqp.reg <- function(x, z, k, maxiter = 1000, tol = 1e-6, ncores = 1) {
  n <- dim(x)[1]  ;  p <- dim(x)[2]  ;  q <- dim(z)[2]

  runtime <- proc.time()

  # Initialize
  med <- Rfast::Median(x)
  B <- matrix( Rfast::Runif(q * p, 0, med), q, p )
  W <- matrix( Rfast2::Runif(n * k, 0, med), n, k )
  H <- matrix( Rfast2::Runif(k * p, 0, med), k, p )

  sse_old <- Inf
  cl <- NULL
  dq <- diag(q)  ;  dk <- diag(k)
  b0q <- numeric(q)  ;  b0k <- numeric(k)
  Dmatz <- crossprod(z)

  # Setup parallel cluster if needed
  if ( ncores > 1 ) {
    cl <- parallel::makeCluster(ncores)
    on.exit( parallel::stopCluster(cl), add = TRUE )
    parallel::clusterExport( cl, "solve.QP", envir = environment(solve.QP) )
  }

  for (iter in 1:maxiter) {

    # Update B
    R <- x - W %*% H
    for ( j in 1:p ) {
      dvec <- crossprod(z, R[, j])
      B[, j] <- quadprog::solve.QP( Dmatz, dvec, dq, b0q )$solution
    }

    # Update W
    R <- x - z %*% B
    Dmat <- tcrossprod(H)
    if  ( parallel ) {
      W <- t( parallel::parSapply( cl, 1:n, function(i) {
           dvec <- H %*% R[i, ]
           quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution
      } ) )
    } else {
      for ( i in 1:n ) {
        dvec <- H %*% R[i, ]
        W[i, ] <- quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution
      }
    }

    # Update H
    Dmat <- crossprod(W)
    for ( j in 1:p ) {
      dvec <- crossprod(W, R[, j])
      H[, j] <- quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution
    }

    # Compute SSE
    fitted <- z %*% B + W %*% H
    sse <- sum( (x - fitted)^2 )
    # Check convergence based on SSE change
    if ( abs(sse - sse_old) < tol ) break
    sse_old <- sse
  }

  runtime <- proc.time() - runtime

  list(B = B, W = W, H = H, fitted = fitted, obj = see, iters = iter, runtime = runtime)
}
