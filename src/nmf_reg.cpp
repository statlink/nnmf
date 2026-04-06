#include <RcppEigen.h>
#include <nnsolve/fnnls_core.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::export]]
Rcpp::List nmf_reg_cpp(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Z,
                       const int low_dim,
                       const Rcpp::Nullable<Eigen::MatrixXd> H_init,
                       const int max_iter, const double tol, const int ncores) {

#ifdef _OPENMP
  if (ncores > 1)
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

  const double ridge = 1e-6;
  const int predictors = X.cols(), observations = X.rows();
  const int covariates = Z.cols();

  Eigen::MatrixXd H(low_dim, predictors);
  Eigen::MatrixXd HHt(low_dim, low_dim);
  if (H_init.isNotNull()) {
    H = Rcpp::as<Eigen::MatrixXd>(H_init);
  } else {
    Rcpp::NumericVector r = Rcpp::runif(low_dim * predictors, 0.0, 1.0);
    for (int j = 0; j < predictors; ++j)
      for (int i = 0; i < low_dim; ++i)
        H(i, j) = r[i + j * low_dim];
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W(
      observations, low_dim);
  Eigen::MatrixXd WtW(low_dim, low_dim);
  Eigen::MatrixXd WtX(low_dim, predictors);

  Eigen::MatrixXd Zt = Z.transpose();
  Eigen::MatrixXd ZtZ = Zt * Z;
  ZtZ.diagonal().array() += ridge;
  Eigen::LDLT<Eigen::MatrixXd> ZtZ_ldlt(ZtZ);

  Eigen::MatrixXd B(covariates, predictors);
  B = ZtZ_ldlt.solve(Zt * X);
  B = B.cwiseMax(0.0);
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
    HHt.diagonal().array() += ridge;

#ifdef _OPENMP
#pragma omp parallel
    {
      Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int i = 0; i < observations; ++i) {
        rhs.noalias() = H * X_new.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int i = 0; i < observations; ++i) {
        rhs.noalias() = H * X_new.row(i).transpose();
        W.row(i) = fnnls_core(HHt, rhs, tol, max_iter).transpose();
      }
    }
#endif

    WtW.setZero();
    WtW.selfadjointView<Eigen::Upper>().rankUpdate(W.transpose());
    WtW = WtW.selfadjointView<Eigen::Upper>();
    WtW.diagonal().array() += ridge;
    WtX.noalias() = W.transpose() * X_new;

#ifdef _OPENMP
#pragma omp parallel
    {
      Eigen::VectorXd rhs(low_dim);
#pragma omp for schedule(dynamic, chunk)
      for (int j = 0; j < predictors; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls_core(WtW, rhs, tol, max_iter);
      }
    }
#else
    {
      Eigen::VectorXd rhs(low_dim);
      for (int j = 0; j < predictors; ++j) {
        rhs = WtX.col(j);
        H.col(j) = fnnls_core(WtW, rhs, tol, max_iter);
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
