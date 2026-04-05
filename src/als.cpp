//[[Rcpp::plugins(openmp)]]
//[[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <nnsolve/fnnls_core.h>
#include <nnsolve/types.h>
#include <cfloat>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

Rcpp::List nmf_als_veo(const Mat &X, const int low_dim,
                       const Rcpp::Nullable<Mat> W_init = R_NilValue,
                       const Rcpp::Nullable<Mat> H_init = R_NilValue,
                       const double lr_h = 0.1, const double tol = 1e-4,
                       const int max_iter = 1000, const bool parallel = false) {

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
  const double lambda = 1e-6;

  Mat H(low_dim, D);

  if (H_init.isNotNull()) {
    H = Rcpp::as<Mat>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * D, 0.0, 1.0);
    for (int j = 0; j < D; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  Mat neg_gradient(low_dim, D);

  RowMajorMat W(n, low_dim);

  if (W_init.isNotNull()) {
    W = Rcpp::as<Mat>(W_init);
  } else {
    W.setZero();
  }

  Mat HHt(low_dim, low_dim);
  Mat WtW(low_dim, low_dim);
  Mat WH(n, D);

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
    HHt.diagonal().array() += lambda;

#ifdef _OPENMP
#pragma omp parallel if (parallel)
    {
      Vec rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)

      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls(HHt, rhs).transpose();
      }
    }
#else
    {
      Vec rhs(low_dim);
      for (int i = 0; i < n; ++i) {
        rhs.noalias() = H * X.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs).transpose();
      }
    }
#endif

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += lambda;

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
Rcpp::List nmf_als(const Eigen::MatrixXd &X, const int low_dim,
                   const Rcpp::Nullable<Eigen::MatrixXd> W_init = R_NilValue,
                   const Rcpp::Nullable<Eigen::MatrixXd> H_init = R_NilValue,
                   const bool veo = false, const double lr_h = 0.1,
                   const double tol = 1e-4, const int max_iter = 1000,
                   const bool parallel = false, const int ncores = -1) {

#ifdef _OPENMP
  if (parallel && ncores > 0)
    omp_set_num_threads(ncores);
#endif

  if (veo) {
    return nmf_als_veo(X, low_dim, W_init, H_init, lr_h, tol, max_iter,
                       parallel);
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

  const int n = X.rows();
  const int D = X.cols();
  const double lambda = 1e-6;

  // -------------------------
  // Initialize H
  // -------------------------
  Eigen::MatrixXd H(low_dim, D);

  if (H_init.isNotNull()) {
    H = Rcpp::as<Eigen::MatrixXd>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * D, 0.0, 1.0);
    for (int j = 0; j < D; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  // -------------------------
  // Initialize W (row-major)
  // -------------------------
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrix;
  RowMajorMatrix W(n, low_dim);

  if (W_init.isNotNull()) {
    W = Rcpp::as<Eigen::MatrixXd>(W_init);
  } else {
    W.setZero();
  }

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
    HHt.diagonal().array() += lambda;

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

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += lambda;

    WtX.noalias() = W.transpose() * X;

#ifdef _OPENMP
#pragma omp parallel if (parallel)
    {
      Vec rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int j = 0; j < D; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls(WtW, rhs);
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int j = 0; j < D; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls(WtW, rhs);
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
