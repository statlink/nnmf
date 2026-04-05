#include <RcppEigen.h>
#include <nnsolve/fnnls.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::export]]
Rcpp::List nmf_reg(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Z,
                   const int low_dim,
                   const Rcpp::Nullable<Eigen::MatrixXd> H_init = R_NilValue,
                   const Rcpp::Nullable<Eigen::MatrixXd> W_init = R_NilValue,
                   const Rcpp::Nullable<Eigen::MatrixXd> B_init = R_NilValue,
                   const int max_iter = 1000, const double tol = 1e-4,
                   const bool parallel = false, const int ncores = -1) {

#ifdef _OPENMP
  if (parallel && ncores > 0)
    omp_set_num_threads(ncores);
#endif

  int chunk = 1;
  if (low_dim <= 8)
    chunk = 32;
  else if (low_dim <= 16)
    chunk = 16;
  else if (low_dim <= 32)
    chunk = 8;
  else
    chunk = 4;

  const double lambda = 1e-6;
  const int predictors = X.cols(), observations = X.rows();
  const int covariates = Z.cols();

  Eigen::MatrixXd H(low_dim, predictors);
  if (H_init.isNotNull()) {
    H = Rcpp::as<Eigen::MatrixXd>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * predictors, 0.0, 1.0);
    for (int j = 0; j < predictors; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  Eigen::Matrix::<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>W(
      observations, low_dim);

  if (W_init.isNotNull()) {
    W = Rcpp::as<Eigen::MatrixXd>(W_init);
  } else {
    W.setZero();
  }

  Eigen::MatrixXd Zt = Z.transpose();
  Eigen::MatrixXd ZtZ = Zt * Z;
  ZtZ.diagonal().array() += lambda;
  Eigen::LDLT<Eigen::MatrixXd> ZtZ_ldlt(ZtZ);

  Eigen::MatrixXd B(covariates, predictors);

  if (B_init.isNotNull()) {
    B = Rcpp::as<Eigen::MatrixXd>(B_init);
  } else {
    B = ZtZ_ldlt.solve(Zt * X);
    B = B.cwiseMax(0.0);
  }

  Eigen::MatrixXd HHt(low_dim, low_dim);
  Eigen::MatrixXd WtW(low_dim, low_dim);
  Eigen::MatrixXd WtX(low_dim, predictors);

  Eigen::MatrixXd X_new = X;
  X_new -= Z * B;

  Eigen::MatrixXd WH(observations, predictors);
  Eigen::MatrixXd ZB(observations, predictors);
  Eigen::MatrixXd fitted(observations, predictors);

  const double Xnorm2 = X.squaredNorm();
  double obj_prev = 1.0, obj_curr = 2.0;
  int iter = 0;

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
      for (int i = 0; i < observations; ++i) {
        rhs.noalias() = H * X_new.row(i).transpose();
        W.row(i) = fnnls(HHt, rhs).transpose();
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int i = 0; i < observations; ++i) {
        rhs.noalias() = H * X_new.row(i).transpose();
        W.row(i) = fnnls(HHt, rhs).transpose();
      }
    }
#endif

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += lambda;

    WtX.noalias() = W.transpose() * X_new;

#ifdef _OPENMP
#pragma omp parallel if (parallel)
    {
      Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int j = 0; j < predictors; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls(WtW, rhs);
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int j = 0; j < predictors; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls(WtW, rhs);
      }
    }
#endif

    WH.noalias() = W * H;

    B.noalias() = ZtZ_ldlt.solve(Zt * (X - WH));

    ZB.noalias() = Z * B;

    X_new = X;
    X_new.noalias() -= ZB;

    fitted.noalias() = ZB + WH;

    const double term2 = 2.0 * (X.cwiseProduct(fitted)).sum();
    const double term3 = fitted.squaredNorm();
    obj_curr = Xnorm2 - term2 + term3;
  }

  return Rcpp::List::create(
      Rcpp::Named("B") = B, Rcpp::Named("W") = W, Rcpp::Named("H") = H,
      Rcpp::Named("fitted") = fitted, Rcpp::Named("obj") = obj_curr,
      Rcpp::Named("iters") = iter);
}
