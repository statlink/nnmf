nmfqp.reg <- function(x, z, k, maxiter = 1000, tol = 1e-6, ncores = 1) {
  n <- dim(x)[1]  ;  p <- dim(x)[2]  ;  q <- dim(z)[2]

  runtime <- proc.time()

  # Initialize
  med <- Rfast::Median(x)
  W <- matrix( rangen::Runif(n * k, 0, med), n, k )
  H <- matrix( rangen::Runif(k * p, 0, med), k, p )
  B <- matrix(nrow = q, ncol = p)

  sse_old <- Inf
  cl <- NULL
  dq <- diag(q)  ;  dk <- diag(k)
  b0q <- numeric(q)  ;  b0k <- numeric(k)
  Dmatz <- crossprod(z)
  w1 <- rep(1/k, k)  ;  h1 <- rep(1/p, p)  ;  b1 <- rep(1/q, q)

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
      B[, j] <- tryCatch( pmax( quadprog::solve.QP( Dmatz, dvec, dq, b0q )$solution, 0 ),
                          error = function(e) { return( b1 ) } )
    }
    # Update W
    R <- x - z %*% B
    Dmat <- tcrossprod(H)
    if  ( ncores > 1 ) {
      W <- t( parallel::parSapply( cl, 1:n, function(i) {
        dvec <- H %*% R[i, ]
        abs( quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution )
      } ) )
    } else {
      for ( i in 1:n ) {
        dvec <- H %*% R[i, ]
        W[i, ] <- tryCatch( pmax( quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution, 0 ),
                            error = function(e) { return( w1 ) } )
      }
    }

    # Update H
    Dmat <- crossprod(W)
    for ( j in 1:p ) {
      dvec <- crossprod(W, R[, j])
      tryCatch(
        H[i, ] <- pmax( quadprog::solve.QP( Dmat, dvec, dk, b0k )$solution, 0 ),
        error = function(e) { return( h1 ) } )
    }

    # Compute SSE
    fitted <- z %*% B + W %*% H
    sse <- sum( (x - fitted)^2 )
    # Check convergence based on SSE change
    if ( abs(sse - sse_old) < tol ) break
    sse_old <- sse
  }

  runtime <- proc.time() - runtime

  list(B = B, W = W, H = H, fitted = fitted, obj = sse, iters = iter, runtime = runtime)
}
