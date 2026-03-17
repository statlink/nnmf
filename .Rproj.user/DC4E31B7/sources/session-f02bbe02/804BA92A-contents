\name{nmfqp.cv}
\alias{nmfqp.cv}
\title{
K-fold cross-validation for choosing the rank in NMF
}

\description{
K-fold cross-validation for choosing the rank in NMF.
}

\usage{
nmfqp.cv(x, k = 3:10, k_means = TRUE, bs = 1, veo = FALSE, lr_h = 0.1, maxiter = 1000,
tol = 1e-6, ridge = 1e-8, ncores = 1, folds = NULL, nfolds = 10, graph = FALSE)
}

\arguments{
\item{x}{
An \eqn{n \times D} matrix with compositional data. Zero values are allowed.
}
\item{k}{
The number of lower dimensions. It must be less than the dimensionality of the data, at most \eqn{D-1}.
}
\item{k_means}{
If this is TRUE, then the K-means algorithm is used to initiate the W and H matrices.
}
\item{bs}{
If you use the K-means algorithm for initialization, you may want to use the mini batch K-means if you have millions of observations. In this case, you need to define the number of batches.
}
\item{veo}{
If the number of variables excceeds the number of observations set this is equal to TRUE.
In this case, the sparse k-means algorithm of Witten and Tibshirani (2010) is used to initialize the H matrix.
}
\item{lr_h}{
If veo is TRUE, then the exponentiated gradient descent method is used to update the H matrix. In this case you need to supply the value of the learning rate, which is 0.1 by default.
}
\item{maxiter}{
The maximum number of iterations allowed.
}
\item{tol}{
The tolerance value to terminate the quadratic programming algorithm.
}
\item{ridge}{
A small quantity added in the diagonal of the \eqn{D} matrix.
}
\item{history}{
If this is TRUE, the reconstruction error at each iteration is returned.
}
\item{ncores}{
Do you want the update of W to be performed in parallel? If yes, specify the number of cores to use.
}
\item{graph}{
If this is TRUE, the plot of the predicted error will be plotted.
}
}

\details{
K-fold cross-validation to select the optimal rank k.
}

\value{
\item{sse}{
The matrix with the sum of squares of residuals.
}
\item{mspe}{
A vector with the mean squares of residuals.
}
\item{runtime}{
The runtime required by the algorithm.
}
}

\references{
Wang Y. X. and Zhang Y. J. (2012). Nonnegative matrix factorization: A comprehensive review. IEEE Transactions on Knowledge and Data Engineering, 25(6): 1336-1353.

Kim H. and Park H. (2008). Nonnegative matrix factorization based on alternating nonnegativity constrained least squares and active set method. SIAM Journal on Matrix Analysis and Applications, 30(2): 713-730.
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{ \link{nmf.qp}, \link{nmfqp.pred}
}
}

\examples{
x <- as.matrix(iris[, 1:4])
est <- snmfqp.pred(x[1:10, ], 3, mod$H)
}

