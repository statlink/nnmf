\name{nmfqp.reg}
\alias{nmfqp.reg}
\title{
NMF with covariates minimizing the Frobenius norm
}

\description{
NMF with covariates minimizing the Frobenius norm using quadratic programming.
}

\usage{
nmfqp.reg(X, Z, k, H_init, maxiter = 1000, tol = 1e-6, ncores = 1)
}

\arguments{
\item{X}{
An \eqn{n \times D} numerical matrix with data.
}
\item{Z}{
An \eqn{n \times q} matrix with the covariates.
}
\item{k}{
The number of lower dimensions. It must be less than the dimensionality of the data, at most \eqn{D-1}.
}
\item{H_init}{
If you have an initial estimate for H supply it here, otherwise leave it NULL.
}
\item{maxiter}{
The maximum number of iterations allowed.
}
\item{tol}{
The tolerance value to terminate the quadratic programming algorithm.
}
\item{ncores}{
Do you want the update of W to be performed in parallel? If yes, specify the number of cores to use.
}
}

\details{
Nonnegative matrix factorization with covariates using quadratic programming is performed.
The objective function to be minimized is the square of the Frobenius norm of the residuals produced by the
reconstructed matrix.
}

\value{
\item{B}{
The \eqn{B} matrix, an \eqn{q \times D} matrix with the coefficients of the covariates.
}
\item{W}{
The \eqn{W} matrix, an \eqn{n \times k} matrix with the mapped data.
}
\item{H}{
The \eqn{H} matrix, an \eqn{k \times D} matrix.
}
\item{fitted}{
The reconstructed data, \eqn{fitted = ZB + WH}.
}
\item{obj}{
The reconstruction error, \eqn{ ||x - fitted||_F^2}.
}
\item{iters}{
The number of iterations performed.
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
\code{ \link{nmf.sqp}
}
}

\examples{
X <- as.matrix(iris[, 1:3])
Z <- model.matrix(X ~., data = iris[, 4:5])[, -1]
mod <- nmfqp.reg(X, Z, 2)
}

