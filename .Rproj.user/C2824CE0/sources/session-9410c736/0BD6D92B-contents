# Dirichlet Component Analysis (DCA)
# Based on Wang et al. (2008)
# Dirichlet Component Analysis (DCA)
# Based on Wang et al. (2008)
# Corrected version with proper R matrix dimensions
dca <- function(x, k, maxiter = 1000, pop_size = 50, tol = 1e-4) {
  runtime <- proc.time()
  M <- dim(x)[1]  ;  N <- dim(x)[2]

  # Initialize population of balanced rearrangement matrices (N x k)
  population <- list()
  for (i in 1:pop_size) {
    population[[i]] <- .random_balanced_rearrangement(N, k)
  }

  best_R <- NULL
  best_alpha <- Inf
  prev_best_alpha <- Inf

  # Genetic algorithm
  for (iter in 1:maxiter) {
    # Evaluate fitness for each individual
    alphas <- numeric(pop_size)
    for (i in 1:pop_size) {
      R <- population[[i]]
      # Apply transformation: x is M×N, R is N×k, result is M×k
      xtrans <- x %*% R
      # Apply regularization
      xreg <- .regularize_data(xtrans)
      # Estimate Dirichlet correlation
      alphas[i] <- .diri_a0(xreg)
    }

    # Track best solution
    min_idx <- which.min(alphas)
    if (alphas[min_idx] < best_alpha) {
      prev_best_alpha <- best_alpha
      best_alpha <- alphas[min_idx]
      best_R <- population[[min_idx]]
    }

    # Check convergence using relative change in best alpha
    if (iter > 10 && is.finite(prev_best_alpha)) {
      relative_change <- abs(best_alpha - prev_best_alpha) / max(abs(prev_best_alpha), 1e-8)
      if (relative_change < tol) {
        break
      }
    }

    # Selection and crossover
    fitness <- .compute_fitness(alphas)
    fitness[fitness == 0] <- 1e-10  # Avoid zero probabilities
    probs <- fitness / sum(fitness)

    # Create next generation
    new_population <- list()
    new_population[[1]] <- best_R  # Elitism

    # Reduce population size
    current_size <- max(10, pop_size - floor(iter / 2))

    for (i in 2:current_size) {
      # Sample two parents
      parents <- sample(1:pop_size, 2, prob = probs, replace = TRUE)
      R1 <- population[[parents[1]]]
      R2 <- population[[parents[2]]]
      # Crossover: weighted average
      weight <- fitness[parents[1]] / (fitness[parents[1]] + fitness[parents[2]])
      R_new <- weight * R1 + (1 - weight) * R2
      new_population[[i]] <- R_new
    }

    population <- new_population
    pop_size <- length(population)
  }

  # Return results
  R <- best_R  # R is N×k
  W <- x %*% R  # W is M×k
  W <- .regularize_data(W)
  runtime <- proc.time() - runtime
  list(W = W, R = R, alpha = best_alpha, iters = iter, runtime = runtime)
}


# Function to estimate Dirichlet precision parameter (alpha_0) using Newton-Raphson
.diri_a0 <- function(x, tol = 1e-6, maxiter = 100) {
  n <- dim(x)[1]  ;  D <- dim(x)[2]

  # Better initialization: method of moments
  m <- Rfast::colmeans(x)
  v <- sum(Rfast::colVars(x))
  a0_init <- max(0.1, m[1] * (m[1] * (1 - m[1]) / v - 1))
  a <- log(a0_init)
  ea <- a0_init

  slx <- sum(Rfast::Log(x))
  lik1 <- n * (lgamma(D * ea) - D * lgamma(ea)) + (ea - 1) * slx

  for (iter in 1:maxiter) {
    grad <- n * D * ea * (digamma(D * ea) - digamma(ea)) + ea * slx
    hess <- n * trigamma(D * ea) * (D * ea)^2 - n * D * trigamma(ea) * ea^2 + grad

    a <- a - grad / hess
    ea <- exp(a)
    lik2 <- n * (lgamma(D * ea) - D * lgamma(ea)) + (ea - 1) * slx

    if (abs(lik2 - lik1) < tol) break
    lik1 <- lik2
  }

  ea
}

# Function to apply regularization operator
.regularize_data <- function(X, epsilon = 1e-10) {
  # X: M x k matrix of compositional data
  # Returns regularized data
  delta <- min(X)
  # Apply regularization as paper defines, but add epsilon for numerical stability
  X_reg <- (X - delta) + epsilon
  # Radial projection back to simplex
  X_reg <- X_reg / Rfast::rowsums(X_reg)
  return(X_reg)
}

# Function to check if matrix is a balanced rearrangement
# R should be N×k where rows sum to 1 and columns sum to k/N
.is_balanced_rearrangement <- function(R, N, K, tol = 1e-6) {
  # Check if R is a matrix
  if (!is.matrix(R)) return(FALSE)

  # Check dimensions safely
  dims <- dim(R)
  if (is.null(dims) || length(dims) != 2) return(FALSE)
  if (dims[1] != N || dims[2] != K) return(FALSE)

  # Check non-negativity
  if (any(R < -tol)) return(FALSE)

  # Check row sums = 1
  row_sums <- Rfast::rowsums(R)
  if (any(abs(row_sums - 1) > tol)) return(FALSE)

  # Check column sums = k/N
  col_sums <- Rfast::colsums(R)
  if (any(abs(col_sums - K/N) > tol)) return(FALSE)

  return(TRUE)
}

# Function to generate random balanced rearrangement matrix
# Returns N×k matrix where rows sum to 1 and columns sum to k/N
.random_balanced_rearrangement <- function(N, K) {
  # Initialize with uniform probabilities (N×k matrix)
  R <- matrix(Rfast2::Runif(N * K), nrow = N, ncol = K)

  # Make it satisfy constraints using iterative proportional fitting
  for (iter in 1:1000) {
    # Normalize rows to sum to 1
    row_sums <- Rfast::rowsums(R)
    R <- R / row_sums

    # Normalize columns to sum to k/N
    col_sums <- Rfast::colsums(R)
    for (j in 1:K) {
      R[, j] <- R[, j] / col_sums[j] * (K / N)
    }

    if (.is_balanced_rearrangement(R, N, K)) break
  }

  R
}

# Fitness function for genetic algorithm
.compute_fitness <- function(alpha_values) {
  median_alpha <- median(alpha_values)
  -log(pmin(alpha_values / median_alpha, 1))
}
