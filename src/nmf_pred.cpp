#include <RcppEigen.h>
#include <nnsolve/fnnls.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::export]]
Rcpp::List nmf_pred(const Eigen::MatrixXd &X, const Eigen::MatrixXd &H,
                    const bool parallel = false, const int ncores = -1) {

#ifdef _OPENMP
  if (parallel && ncores > 0)
    omp_set_num_threads(ncores);
#endif

  const int low_dim = H.rows();
  const int n = X.rows();
  const double lambda = 1e-7;

  int chunk = 1;
  if (low_dim <= 8)
    chunk = 32;
  else if (low_dim <= 16)
    chunk = 16;
  else if (low_dim <= 32)
    chunk = 8;
  else
    chunk = 4;

  Eigen::MatrixXd HHt(low_dim, low_dim);
  HHt.setZero();
  HHt.selfadjointView<Eigen::Upper>().rankUpdate(H);
  HHt = HHt.selfadjointView<Eigen::Upper>();
  HHt.diagonal().array() += lambda;

  Eigen::Matrix::<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W(n, low_dim);

#ifdef _OPENMP
#pragma omp parallel if (parallel)
  {
    Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
    for (int i = 0; i < n; ++i) {
      rhs.noalias() = H * X.row(i).transpose();
      W.row(i) = fnnls(HHt, rhs).transpose();
    }
  }
#else
  {
    Eigen::VectorXd rhs(low_dim);
    for (int i = 0; i < n; ++i) {
      rhs.noalias() = H * X.row(i).transpose();
      W.row(i) = fnnls(HHt, rhs).transpose();
    }
  }
#endif

  return Rcpp::List::create(Rcpp::Named("Wnew") = W,
                            Rcpp::Named("Znew") = W * H);
}
