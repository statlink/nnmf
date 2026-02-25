\name{nmfqp.reg}
\alias{nmfqp.reg}
\title{
NMF with covariates minimizing the Frobenius norm
}

\description{
NMF with covariates minimizing the Frobenius norm using quadratic programming.
}

\usage{
nmfqp.reg(x, z, k, maxiter = 1000, tol = 1e-6, ncores = 1)
}

\arguments{
\item{x}{
An \eqn{n \times D} numerical matrix with data.
}
\item{z}{
A matrix with covariates.
}
\item{k}{
The number of lower dimensions. It must be less than the dimensionality of the data, at most \eqn{D-1}.
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
The \eqn{B} matrix, the coefficients of the covariates.
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
Alenazi A. and Tsagris M. (2026). Simplicial nonnegative matrix factorization. In preparation.

Cutler A. and Breiman L. (1994). Archetypal analysis. Technometrics, 36(4): 338--347.
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
x <- as.matrix(iris[, 1:3])
z <- model.matrix(x ~., data = iris[, 4:5])[, 1]
mod <- nmfqp.reg(x, z, 2)
}

