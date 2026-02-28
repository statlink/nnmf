# Define helper OUTSIDE the main function to avoid closure capture
.solve_w_row_qp <- function(i, G_W, g_W, A_W, b_W) {
  sol <- quadprog::solve.QP(Dmat = G_W, dvec = g_W[, i], Amat = A_W, bvec = b_W, meq = 0)
  abs(sol$solution)
}

nmf.qp <- function(x, k, W = NULL, H = NULL, k_means = TRUE, bs = 1, veo = FALSE, lr_h = 0.1,
                    maxiter = 1000, tol = 1e-6, ridge = 1e-8, history = FALSE, ncores = 1) {

  runtime <- proc.time()
  n <- dim(x)[1]  ;  D <- dim(x)[2]

  # Initialize W, H with non-negativity only (no simplex)
  if ( k_means ) {
   a <- nnmf::init(x, k, bs, veo)
   W <- a$W   ;  H <- a$H
  } else {
    if ( is.null(W) ) {
      W <- matrix( Rfast2::Runif(n * k, 0, Rfast::Median(x)), nrow = n, ncol = k )
    }
    if ( is.null(H) ) {
      H <- matrix( Rfast2::Runif(k * D, 0, 10), nrow = k, ncol = D )
    }
  }

  error <- numeric(maxiter)
  # Only non-negativity constraints: W >= 0
  A_W <- diag(k)
  b_W <- rep(0, k)
  ridgek <- diag(ridge, k)

  suppressWarnings({

  # CREATE CLUSTER ONCE BEFORE LOOP
  if (ncores > 1) {
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, library(quadprog))
    parallel::clusterExport( cl, varlist = c(".solve_w_row_qp", "A_W", "b_W"), envir = environment() )
  }

  if ( !veo ) {  ## veo is FALSE, n > p

    sx2 <- sum(x^2)
    kD <- k * D
    # Only non-negativity constraints: H >= 0
    A <- diag(kD)
    bvec <- rep(0, kD)

    for (it in 1:maxiter) {
      # ----------------- W update -----------------
      G_W <- 2 * tcrossprod(H) + ridgek
      g_W <- 2 * tcrossprod(H, x)

      if (ncores > 1) {
        # Export iteration-specific variables
        parallel::clusterExport(cl, varlist = c("G_W", "g_W"), envir = environment())
        suppressWarnings({
          W <- t( parallel::parSapply(cl, 1:n, .solve_w_row_qp, G_W = G_W, g_W = g_W, A_W = A_W, b_W = b_W) )
        })
      } else {
        for (i in 1:n) {
          sol <- quadprog::solve.QP(Dmat = G_W, dvec = g_W[, i], Amat = A_W, bvec = b_W, meq = 0)
          W[i, ] <- abs(sol$solution)
        }
      }

      # ----------------- H update -----------------
      dvec <- as.vector( crossprod(W, x) )
      XX <- kronecker(diag(D), crossprod(W))
      f <- try(quadprog::solve.QP(Dmat = XX, dvec = dvec, Amat = A, bvec = bvec, meq = 0), silent = TRUE)
      if (identical(class(f), "try-error")) {
        f <- quadprog::solve.QP(Dmat = Matrix::nearPD(XX)$mat, dvec = dvec, Amat = A, bvec = bvec, meq = 0)
      }
      H <- matrix(abs(f$solution), nrow = k, ncol = D)
      err <- sx2 + 2 * f$value
      error[it] <- err

      if (it > 1 && abs(error[it - 1] - err) < tol) {
        break
      }
    }
	Z <- W %*% H

  } else {  ## veo is TRUE, n < p

    sx2 <- sum(x^2)

    for (it in 1:maxiter) {
      # ----------------- W update -----------------
      G_W <- 2 * tcrossprod(H) + ridgek
      g_W <- 2 * tcrossprod(H, x)

      if (ncores > 1) {
        # Export iteration-specific variables
        parallel::clusterExport(cl, varlist = c("G_W", "g_W"), envir = environment())
        suppressWarnings({
          W <- t( parallel::parSapply(cl, 1:n, .solve_w_row_qp, G_W = G_W, g_W = g_W, A_W = A_W, b_W = b_W) )
        })
      } else {
        for (i in 1:n) {
          sol <- quadprog::solve.QP(Dmat = G_W, dvec = g_W[, i], Amat = A_W, bvec = b_W, meq = 0)
          W[i, ] <- abs(sol$solution)
        }
      }

      # ----------------- H update -----------------
      # Exponentiated gradient WITHOUT simplex normalization
      E <- W %*% H - x
      grad_h <- crossprod(W, E)
      H <- H * exp(-lr_h * grad_h)
      # NO normalization - keep only non-negativity
      Z <- W %*% H
      err <- sum( (x - Z)^2 )
      error[it] <- err

      if (it > 1 && abs(error[it - 1] - err) < tol) {
        break
      }
    }

  }  ##  end if (!veo)

  })  # end suppressWarnings

  runtime <- proc.time() - runtime
  error <- error[1:it]
  obj <- error[it]
  if ( !history )  error <- NULL
  colnames(H) <- colnames(x)
  
  list(W = W, H = H, Z = Z, obj = obj, error = error, iters = it, runtime = runtime)
}
