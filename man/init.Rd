\name{init}
\alias{init}
\title{
Initialization strategies for the NMF  based on the k-means
}

\description{
Initialization strategies for the NMF based on the k-means algorithm.
}

\usage{
init(x, k, bs = 1, veo = FALSE)
}

\arguments{
\item{x}{
An \eqn{n \times D} numerical matrix with data.
}
\item{k}{
The number of lower dimensions. It must be less than the dimensionality of the data, at most \eqn{D-1}.
}
\item{bs}{
The batch size in case the user wants to use the mini-batch k-means algorithm. If bs=1, the classical k-means is used.
}
\item{veo}{
If the (number of) variables exceed the (number of) observations set this equal to true. In this case, the sparse k-means algorithm
of Witten and Tibshirani (2010) is used to initiate the H matrix.
}
}

\details{
Nonnegative matrix factorization using quadratic programming is performed. 
The objective function to be minimized is the square of the Frobenius norm.
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
Alenazi A. and Tsagris M. (2026). Simplicial nonnegative matrix factorization. In preparation.

Witten D. M. and Tibshirani R. (2010). A framework for feature selection in clustering. 
Journal of the American Statistical Association, 105(490): 713--726.

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
\code{ \link{nmf.qp}
}
}

\examples{
x <- as.matrix(iris[, 1:4])
mod <- nmf.qp(x, 2)
plot(mod$W, colour = iris[, 5])
}

