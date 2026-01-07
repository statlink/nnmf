################################################################################
# Manhattan NMF via Linear Programming (Rglpk)
# L1 minimization for general non-negative data
################################################################################

# Define helper OUTSIDE the main function to avoid closure capture
.solve_w_row_lp <- function(i, Ht, x, obj_vec, mat_base, dir_vec, bounds_list) {
  n <- nrow(Ht)
  p <- ncol(Ht)
  y <- x[i, ]
  # Build full constraint matrix for this row
  mat <- rbind( cbind(Ht, diag(n) ),      # First n constraints
                cbind(-Ht, diag(n) ) )     # Second n constraints
  rhs <- c(y, -y)
  # Solve LP
  result <- Rglpk::Rglpk_solve_LP( obj = obj_vec, mat = mat, 
            dir = dir_vec, rhs = rhs, bounds = bounds_list, max = FALSE )
  result$solution[1:p]
}


#' Manhattan NMF using Linear Programming
#' 
#' @param x Non-negative matrix to factorize (m x n)
#' @param k Number of components
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @param ncores Number of cores for parallel processing (default: 1, no parallel)
#' @param verbose Print progress
#' @return List with W, H matrices and convergence info
nmf.manh <- function(x, k, max_iter = 100, tol = 1e-6, ncores = 1) {
  
  runtime <- proc.time()
  m <- nrow(x)  ;  n <- ncol(x)
  # Initialize W and H
  set.seed(123)
  W <- matrix( Rfast2::Rrunif(m * k, 0, Rfast::Median(x)), m, k )
  H <- matrix( Rfast2::Runif(k * n, 0, 10), k, n )
  
  obj_history <- numeric(max_iter)
  
  # Fixed LP problem structure
  obj_vec <- c(rep(0, k), rep(1, n))
  dir_vec <- rep(">=", 2*n)
  bounds_list <- list(
    lower = list(ind = 1:(k + n), val = rep(0, k + n)),
    upper = list(ind = 1:(k + n), val = rep(Inf, k + n))
  )
  
  # CREATE CLUSTER ONCE BEFORE LOOP
  if ( ncores > 1 ) {
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, library(Rglpk))
    parallel::clusterExport(cl, 
      varlist = c("obj_vec", "dir_vec", "bounds_list", ".solve_w_row_lp"), 
      envir = environment())
  }
  
  for ( iter in 1:max_iter ) {
    
    # Update H (fixing W) - sequential
    for ( j in 1:n ) {
      p <- ncol(W)
      nn <- nrow(W)
      y <- x[, j]
      
      obj <- c(rep(0, p), rep(1, nn))
      mat <- rbind( cbind(W, diag(nn) ),
                    cbind(-W, diag(nn) ) )
      dir <- rep(">=", 2*nn)
      rhs <- c(y, -y)
      bounds <- list(
        lower = list(ind = 1:(p + nn), val = rep(0, p + nn)),
        upper = list(ind = 1:(p + nn), val = rep(Inf, p + nn))
      )
      result <- Rglpk::Rglpk_solve_LP(obj = obj, mat = mat, dir = dir, rhs = rhs, bounds = bounds, max = FALSE)
      H[, j] <- result$solution[1:p]
    }
    
    # Update W (fixing H) - optionally parallel
    Ht <- t(H)
    if ( ncores > 1 ) {
      parallel::clusterExport(cl, varlist = c("Ht", "V"), envir = environment())
      suppressWarnings({
        W <- t(parallel::parSapply(cl, 1:m, .solve_w_row_lp, Ht = Ht, x = x, obj_vec = obj_vec, 
                                   mat_base = NULL, dir_vec = dir_vec,  bounds_list = bounds_list))
      })
    } else {
      for (i in 1:m) {
        W[i, ] <- .solve_w_row_lp(i, Ht, x, obj_vec, NULL, dir_vec, bounds_list)
      }
    }
    # Compute objective (L1 norm)
	Z <- W %*% H
    res <- x - Z
    obj <- sum( abs(res) )
    obj_history[iter] <- obj 
    # Check convergence
    if (iter > 1) {
      rel_change <- abs( obj_history[iter] - obj_history[iter-1] ) / (obj_history[iter-1] + 1e-10)
      if ( rel_change < tol ) {
        obj_history <- obj_history[1:iter]
        break
      }
    }
  }
  
  runtime <- proc.time() - runtime
  
  list( W = W, H = H, Z = Z, obj = obj_history[iter], iters = iter, runtime = runtime )
}