\name{nmf.hals}
\alias{nmf.hals}
\title{
NMF minimizing using the hierarchical ALS algorithm
}

\description{
NMF minimizing using the hierarchical ALS algorithm.
}

\usage{
nmf.hals(x, k, maxiter = 2000, tol = 1e-6, history = FALSE)
}

\arguments{
\item{x}{
An \eqn{n \times D} numerical matrix with data.
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
\item{history}{
If this is TRUE, the reconstruction error at each iteration is returned.
}
}

\details{
Nonnegative matrix factorization using the hierarchical alternating least squares algorithm
is performed. The objective function to be minimized is the square of the Frobenius norm.
}

\value{
\item{W}{
The \eqn{W} matrix, an \eqn{n \times k} matrix with the mapped data.
}
\item{H}{
The \eqn{H} matrix, an \eqn{k \times D} matrix.
}
\item{Z}{
The reconstructed data, \eqn{Z = WH}.
}
\item{obj}{
The reconstruction error, \eqn{ ||x - Z||_F^2}.
}
\item{error}{
If the argument history was set to TRUE the reconstruction error at each iteration will be performed, otherwise this is NULL.
}
\item{iters}{
The number of iterations performed.
}
\item{runtime}{
The runtime required by the algorithm.
}
}

\references{
Erichson N. B., Mendible A., Wihlborn S. and Kutz J. N. (2018). Randomized nonnegative matrix factorization.
Pattern Recognition Letters, 104: 1-7.
https://arxiv.org/pdf/1711.02037
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{ \link{nmf.qp}, \link{nmf.sqp}
}
}

\examples{
x <- as.matrix(iris[, 1:4])
mod <- nmf.qp(x, 2)
group <- as.numeric(iris[, 5])
plot(mod$W, col = group)
}

