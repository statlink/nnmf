init <- function(x, k, bs = 1, veo = FALSE) {
  
  n <- dim(x)[1]  ;  D <- dim(x)[2]
  if ( bs == 1 ) {
    a <- kmeans( t(x), k)
    W <- matrix(nrow = n, ncol = k)
    for ( i in 1:k )  W[, i] <- Rfast::rowsums(x[, a$cl == i, drop = FALSE])
    if ( veo ) {
      H <- matrix(nrow = k, ncol = D)
      cs <- as.numeric( sparcl::KMeansSparseCluster(x, k, wbounds = 2)[[1]]$Cs )
      for ( i in 1:k )  H[i, ] <- Rfast::colmeans(x[cs == i, , drop = FALSE])
    } else  H <- kmeans(x, k)$centers
    
  } else {
    W <- matrix(nrow = n, ncol = k)
    cs <- as.numeric( sparcl::KMeansSparseCluster(t(x), k, wbounds = 2)[[1]]$Cs )
    for ( i in 1:k )  W[, i] <- Rfast::rowsums(x[, cs == i, drop = FALSE])
    H <- ClusterR::MiniBatchKmeans(x, clusters = k, batch_size = bs)$centroids
  }

  list(W = W, H = H) 
}  
  
