#include <Rcpp.h>
#include <nnsolve/fnnls_core.h>
using namespace Rcpp;

// // [[Rcpp::plugins(openmp)]]
//[[Rcpp::depends(RcppEigen)]]

#include <cfloat>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppEigen.h>
#include <vector>

// Define a RESTRICT macro for portability.
#if defined(__GNUG__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

// -----------------------------------------------------------------------------
// Optimized fnnls (unchanged)
// -----------------------------------------------------------------------------
Eigen::VectorXd fnnls(const Eigen::MatrixXd &XtX, const Eigen::VectorXd &Xty,
                      const double tol = 1e-6, const int max_iter = 1000) {
  const int k = Xty.rows();

  Eigen::VectorXd w = Eigen::VectorXd::Zero(k);
  Eigen::VectorXd negative_gradient = Xty;

  std::vector<char> is_active(k, 1);
  std::vector<int> passive, updated_passive;
  passive.reserve(k);
  updated_passive.reserve(k);

  Eigen::MatrixXd XtX_sub;
  Eigen::VectorXd Xty_sub(k);
  Eigen::VectorXd s_p(k);
  Eigen::VectorXd w_p(k);

  Eigen::LLT<Eigen::MatrixXd> llt;

  double * RESTRICT w_ptr = w.data();

  int outer_iter = 0;
  while (outer_iter < max_iter) {
    ++outer_iter;

    bool optimal = true;
    passive.clear();

    for (int i = 0; i < k; ++i) {
      if (is_active[i] && negative_gradient(i) > tol) {
        passive.push_back(i);
        is_active[i] = 0;
        optimal = false;
      }
    }

    if (optimal) return w;

    int inner_iter = 0;
    while (inner_iter < max_iter) {
      ++inner_iter;

      const int p = passive.size();
      if (p == 0) break;

      XtX_sub.resize(p, p);

      double * RESTRICT XtX_sub_ptr = XtX_sub.data();
      const double * RESTRICT XtX_ptr = XtX.data();
      double * RESTRICT Xty_sub_ptr = Xty_sub.data();
      const double * RESTRICT Xty_ptr = Xty.data();
      const int * RESTRICT passive_ptr = passive.data();

      for (int a = 0; a < p; ++a) {
        const int ia = passive_ptr[a];
        Xty_sub_ptr[a] = Xty_ptr[ia];

        for (int b = a; b < p; ++b) {
          const int ib = passive_ptr[b];
          const double val = XtX_ptr[ia + ib * k];
          XtX_sub_ptr[a + b * p] = val;
          XtX_sub_ptr[b + a * p] = val;
        }
      }

      llt.compute(XtX_sub);
      Eigen::VectorBlock<Eigen::VectorXd> s_p_view = s_p.head(p);
      s_p_view = llt.solve(Xty_sub.head(p));

      bool feasible_sp = true;
      for (int r = 0; r < p; ++r)
        if (s_p_view[r] <= tol) feasible_sp = false;

        if (feasible_sp) {
          const int * RESTRICT passive_ptr2 = passive.data();
          for (int r = 0; r < p; ++r)
            w_ptr[ passive_ptr2[r] ] = s_p_view[r];
          break;
        }

        Eigen::VectorBlock<Eigen::VectorXd> w_p_view = w_p.head(p);
        const int * RESTRICT passive_ptr3 = passive.data();
        for (int r = 0; r < p; ++r)
          w_p_view[r] = w_ptr[ passive_ptr3[r] ];

        double min_a = 2.0;
        for (int r = 0; r < p; ++r) {
          if (s_p_view[r] <= tol) {
            const double denom = w_p_view[r] - s_p_view[r];
            if (denom != 0.0) {
              const double a = w_p_view[r] / denom;
              if (a < min_a) min_a = a;
            }
          }
        }

        updated_passive.clear();
        const int * RESTRICT passive_ptr4 = passive.data();

        if (min_a <= tol) {
          for (int r = 0; r < p; ++r) {
            const int idx = passive_ptr4[r];
            if (s_p_view[r] > tol) {
              w_ptr[idx] = s_p_view[r];
              updated_passive.push_back(idx);
            } else {
              w_ptr[idx] = 0.0;
              is_active[idx] = 1;
            }
          }
        } else {
          for (int r = 0; r < p; ++r) {
            const int idx = passive_ptr4[r];
            const double val = w_p_view[r] + min_a * (s_p_view[r] - w_p_view[r]);
            if (val > tol) {
              w_ptr[idx] = val;
              updated_passive.push_back(idx);
            } else {
              w_ptr[idx] = 0.0;
              is_active[idx] = 1;
            }
          }
        }

        passive.swap(updated_passive);
    }

    negative_gradient = Xty - XtX * w;
  }

  return w;
}


// -----------------------------------------------------------------------------
// nmf_als with optional W_init and H_init
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List nmf_als(const Eigen::MatrixXd &X,
                   const int low_dim,
                   const Rcpp::Nullable<Eigen::MatrixXd> W_init = R_NilValue,
                   const Rcpp::Nullable<Eigen::MatrixXd> H_init = R_NilValue,
                   const double tol = 1e-4,
                   const int max_iter = 1000,
                   const bool parallel = false,
                   const int ncores = -1)
{
#ifdef _OPENMP
  if (parallel && ncores > 0)
    omp_set_num_threads(ncores);
#endif

  int chunk = 1;
  if (low_dim <= 8)      chunk = 32;
  else if (low_dim <= 16) chunk = 16;
  else if (low_dim <= 32) chunk = 8;
  else                    chunk = 4;

  const int n = X.rows();
  const int D = X.cols();
  const double lambda = 1e-6;

  Eigen::MatrixXd ridge = lambda * Eigen::MatrixXd::Identity(low_dim, low_dim);

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
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;
  RowMajorMatrix W(n, low_dim);

  if (W_init.isNotNull()) {
    W = Rcpp::as<Eigen::MatrixXd>(W_init);
  } else {
    W.setZero();
  }

  // -------------------------
  // Rest of the optimized ALS loop (unchanged)
  // -------------------------

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
    HHt += ridge;

#ifdef _OPENMP
#pragma omp parallel if(parallel)
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
WtW += ridge;

WtX.noalias() = W.transpose() * X;

#ifdef _OPENMP
#pragma omp parallel if(parallel)
{
  Eigen::VectorXd rhs(low_dim);
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
    Rcpp::Named("W") = W,
    Rcpp::Named("H") = H,
    Rcpp::Named("Z") = WH,
    Rcpp::Named("obj") = obj_curr,
    Rcpp::Named("iters") = iter
  );
}
