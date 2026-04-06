//[[Rcpp::plugins(openmp)]]
//[[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <cmath>
#include <nnsolve/fnnls_core.h>
#include <nnsolve/types.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

Rcpp::List nmf_als_veo(const Mat &X, const int low_dim,
                       const Rcpp::Nullable<Mat> H_init, const double lr_h,
                       const double tol, const int max_iter,
                       const double ridge) {
  int chunk = 1;
  if (low_dim <= 8)
    chunk = 32;
  else if (low_dim <= 16)
    chunk = 16;
  else if (low_dim <= 32)
    chunk = 8;
  else
    chunk = 4;
  const int n = X.rows();
  const int D = X.cols();

  Mat H(low_dim, D);
  Mat HHt(low_dim, low_dim);
  if (H_init.isNotNull()) {
    H = Rcpp::as<Mat>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * D, 0.0, 1.0);
    for (int j = 0; j < D; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  RowMajorMat W(n, low_dim);
  Mat WtW(low_dim, low_dim);
  Mat WH(n, D);
  Vec neg_gradient(low_dim, D);

  double obj_prev = 1.0, obj_curr = 2.0;
  int iter = 0;
  const double Xnorm2 = X.squaredNorm();
  while (std::abs(obj_prev - obj_curr) / std::abs(obj_prev) > tol &&
         iter < max_iter) {
    ++iter;
    obj_prev = obj_curr;

    HHt.setZero();
    HHt.selfadjointView<Eigen::Upper>().rankUpdate(H);
    HHt = HHt.selfadjointView<Eigen::Upper>();
    HHt.diagonal().array() += ridge;

#ifdef _OPENMP
#pragma omp parallel
    {
      Vec rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)

      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#else
    {
      Vec rhs(low_dim);
      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#endif

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += ridge;

    neg_gradient.noalias() = W.transpose() * X;
    neg_gradient.noalias() -= WtW * H;

    H.array() *= (neg_gradient.array() * lr_h).exp();

    WH.noalias() = W * H;

    const double term1 = Xnorm2;
    const double term2 = 2.0 * (X.cwiseProduct(WH)).sum();
    const double term3 = WH.squaredNorm();
    obj_curr = term1 - term2 + term3;
  }
  return Rcpp::List::create(
      Rcpp::Named("W") = W, Rcpp::Named("H") = H, Rcpp::Named("Z") = WH,
      Rcpp::Named("obj") = obj_curr, Rcpp::Named("iters") = iter);
}

// [[Rcpp::export]]
Rcpp::List nmf_als_cpp(const Eigen::MatrixXd &X, const int low_dim,
                       const Rcpp::Nullable<Eigen::MatrixXd> H_init,
                       const double lr_h, const double tol, const int max_iter,
                       const double ridge, const int ncores) {
#ifdef _OPENMP
  if (ncores > 1)
    omp_set_num_threads(ncores);
#endif

  const int n = X.rows();
  const int D = X.cols();

  if (D >= n) {
    return nmf_als_veo(X, low_dim, H_init, lr_h, tol, max_iter, ridge);
  }

  int chunk = 1;
  if (low_dim <= 8)
    chunk = 32;
  else if (low_dim <= 16)
    chunk = 16;
  else if (low_dim <= 32)
    chunk = 8;
  else
    chunk = 4;

  Eigen::MatrixXd H(low_dim, D);
  if (H_init.isNotNull()) {
    H = Rcpp::as<Eigen::MatrixXd>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * D, 0.0, 1.0);
    for (int j = 0; j < D; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrix;
  RowMajorMatrix W(n, low_dim);

  Eigen::MatrixXd HHt(low_dim, low_dim);
  Eigen::MatrixXd WtW(low_dim, low_dim);
  Eigen::MatrixXd WtX(low_dim, D);
  Eigen::MatrixXd WH(n, D);

  double obj_prev = 1.0, obj_curr = 2.0;
  int iter = 0;
  const double Xnorm2 = X.squaredNorm();
  while (std::abs(obj_prev - obj_curr) / std::abs(obj_prev) > tol &&
         iter < max_iter) {
    ++iter;
    obj_prev = obj_curr;
    HHt.setZero();
    HHt.selfadjointView<Eigen::Upper>().rankUpdate(H);
    HHt = HHt.selfadjointView<Eigen::Upper>();
    HHt.diagonal().array() += ridge;

#ifdef _OPENMP
#pragma omp parallel
    {
      Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#endif

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += ridge;
    WtX.noalias() = W.transpose() * X;

#ifdef _OPENMP
#pragma omp parallel
    {
      Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int j = 0; j < D; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls_core(WtW, rhs, tol, max_iter);
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int j = 0; j < D; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls_core(WtW, rhs, tol, max_iter);
      }
    }
#endif

    WH.noalias() = W * H;
    const double term1 = Xnorm2;
    const double term2 = 2.0 * (X.cwiseProduct(WH)).sum();
    const double term3 = WH.squaredNorm();
    obj_curr = term1 - term2 + term3;
  }
  return Rcpp::List::create(
      Rcpp::Named("W") = W, Rcpp::Named("H") = H, Rcpp::Named("Z") = WH,
      Rcpp::Named("obj") = obj_curr, Rcpp::Named("iters") = iter);
}
