#include <Eigen/Dense>
#include <algorithm>
#include <nnsolve/fnnls_core_help.h>
#include <nnsolve/types.h>
#include <vector>

using std::vector;

bool add_kkt_violators(const Vec &neg_gradient, vector<char> &is_active,
                       vector<int> &passive, const int ncoef,
                       const double tol) {
  const int original_size = passive.size();
  const double *neg_grad_ptr = neg_gradient.data();
  char *is_active_ptr = is_active.data();
  for (int c = 0; c < ncoef; ++c) {
    if (is_active_ptr[c] && neg_grad_ptr[c] > tol) {
      passive.emplace_back(c);
      is_active_ptr[c] = 0;
    }
  }

  return (original_size == passive.size());
}

void grab_passive_idx(const Mat &XtX, const Vec &Xty, Mat &XtX_pass,
                      Vec &Xty_pass, const int *passive_ptr, const int npass,
                      const int ncoef) {
  XtX_pass.resize(npass, npass);
  Xty_pass.resize(npass);

  const double *XtX_ptr = XtX.data();
  const double *Xty_ptr = Xty.data();
  double *XtX_pass_ptr = XtX_pass.data();
  double *Xty_pass_ptr = Xty_pass.data();

  for (int col = 0; col < npass; ++col) {
    const int old_col = passive_ptr[col];
    const int old_offset = old_col * ncoef;
    const int offset = col * npass;

    Xty_pass_ptr[col] = Xty_ptr[old_col];
    XtX_pass_ptr[col + offset] = XtX_ptr[old_col + old_offset];

    for (int row = 0; row < col; ++row) {
      XtX_pass_ptr[row + offset] = XtX_ptr[passive_ptr[row] + old_offset];
    }
  }
}

void split_by_feasibility(const Vec &passive_soln, vector<int> &feasible_idx,
                          vector<int> &infeasible_idx, const double tol,
                          const int npass) {
  feasible_idx.clear();
  infeasible_idx.clear();

  const double *passive_soln_ptr = passive_soln.data();

  for (int i = 0; i < npass; ++i) {
    if (passive_soln_ptr[i] <= tol) {
      infeasible_idx.emplace_back(i);
    } else {
      feasible_idx.emplace_back(i);
    }
  }
}

void update_passive_coeffs(Vec &coeffs, const int *passive_ptr,
                           const Vec &passive_soln, const int npass) {

  double *coeffs_ptr = coeffs.data();
  const double *soln_ptr = passive_soln.data();

  for (int i = 0; i < npass; ++i) {
    coeffs_ptr[passive_ptr[i]] = soln_ptr[i];
  }
}

void extract_passive_coeffs(const Vec &coeffs, Vec &passive_coeffs,
                            const int *passive_ptr, const int npass) {

  passive_coeffs.resize(npass);
  double *pass_coeffs_ptr = passive_coeffs.data();

  const double *coeffs_ptr = coeffs.data();

  for (int i = 0; i < npass; ++i) {
    pass_coeffs_ptr[i] = coeffs_ptr[passive_ptr[i]];
  }
}

double compute_alpha_max(const Vec &passive_coeffs, const Vec &passive_soln,
                         const vector<int> &infeasible_idx) {

  const double *passive_coeffs_ptr = passive_coeffs.data();
  const double *passive_soln_ptr = passive_soln.data();

  double alpha = 2.0;
  for (int infis_idx : infeasible_idx) {
    const double pass_coeff = passive_coeffs_ptr[infis_idx];
    const double a = pass_coeff / (pass_coeff - passive_soln_ptr[infis_idx]);
    alpha = std::min(alpha, a);
  }

  return alpha;
}

void update_coeffs_manual(Vec &coeffs, const Vec &passive_soln,
                          const int *passive_ptr, vector<int> &updated_passive,
                          vector<char> &is_active,
                          const vector<int> &infeasible_idx,
                          const vector<int> &feasible_idx) {

  updated_passive.clear();

  const double *passive_soln_ptr = passive_soln.data();
  double *coeffs_ptr = coeffs.data();
  char *is_active_ptr = is_active.data();

  for (int infis_idx : infeasible_idx) {
    const int new_active_idx = passive_ptr[infis_idx];
    is_active_ptr[new_active_idx] = 1;
    coeffs_ptr[new_active_idx] = 0.0;
  }

  for (int feas_idx : feasible_idx) {
    const int new_passive_idx = passive_ptr[feas_idx];
    updated_passive.emplace_back(new_passive_idx);
    coeffs_ptr[new_passive_idx] = passive_soln_ptr[feas_idx];
  }
}

void update_active_after_step(Vec &coeffs, const vector<int> &passive,
                              vector<int> &updated_passive,
                              vector<char> &is_active, const double tol) {

  updated_passive.clear();
  double *coeffs_ptr = coeffs.data();
  char *is_active_ptr = is_active.data();

  for (int p : passive) {
    if (coeffs_ptr[p] > tol) {
      updated_passive.emplace_back(p);
    } else {
      is_active_ptr[p] = 1;
      coeffs_ptr[p] = 0.0;
    }
  }
}

