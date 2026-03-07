\name{nmfqp.pred}
\alias{nmfqp.pred}
\title{
Prediction of new values using NMF
}

\description{
Prediction of new values using NMF.
}

\usage{
nmfqp.pred(xnew, H, ridge = 1e-8, ncores = 1)
}

\arguments{
\item{xnew}{
An \eqn{n \times D} numerical matrix with new data.
}
\item{H}{
The H matrix produced by the NMF on the observed data.
}
\item{ridge}{
A small quantity added in the diagonal of the \eqn{D} matrix.
}
\item{ncores}{
Do you want the update of W to be performed in parallel? If yes, specify the number of cores to use.
}
}

\details{
Prediction of new values for the nonnegative matrix factorization.
}

\value{
\item{Wnew}{
The \eqn{W} matrix for the new data, an \eqn{n \times k} matrix with the mapped data.
}
\item{Znew}{
The reconstructed new data, \eqn{Znew = WnewHnew}.
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
x <- as.matrix(iris[1:140, 1:4])
xnew <- as.matrix(iris[141:150, 1:4])
mod <- nmf.qp(x, 2)
pred <- nmfqp.pred(xnew, mod$H)
}

