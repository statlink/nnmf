#include <nnsolve/fnnls_core.h>
#include <nnsolve/fnnls_core_help.h>
#include <nnsolve/types.h>
#include <vector>

using std::vector;


Vec fnnls_core(const Mat &XtX, const Vec &Xty, const double tol,
               const int max_iter) {

  const int ncoef = Xty.rows();
  Vec coeffs = Vec::Zero(ncoef), neg_gradient = Xty;

  vector<char> is_active(ncoef, 1);
  vector<int> passive, updated_passive;
  passive.reserve(ncoef);
  updated_passive.reserve(ncoef);

  Eigen::LLT<Mat> llt;

  Mat XtX_pass(ncoef, ncoef);
  Vec Xty_pass(ncoef);

  Vec passive_soln(ncoef), passive_coeffs(ncoef), stepped_soln(ncoef);
  vector<int> feasible_idx, infeasible_idx;
  feasible_idx.reserve(ncoef);
  infeasible_idx.reserve(ncoef);

  int outer_iter = 0;
  while (outer_iter < max_iter) {
    ++outer_iter;

    if (add_kkt_violators(neg_gradient, is_active, passive, ncoef, tol)) {
      return coeffs;
    }

    int inner_iter = 0;
    while (inner_iter < max_iter) {
      ++inner_iter;

      const int *passive_ptr = passive.data();
      const int npass = passive.size();

      grab_passive_idx(XtX, Xty, XtX_pass, Xty_pass, passive_ptr, npass, ncoef);

      llt.compute(XtX_pass.selfadjointView<Eigen::Upper>());
      passive_soln = llt.solve(Xty_pass);

      split_by_feasibility(passive_soln, feasible_idx, infeasible_idx, tol,
                           npass);

      if (infeasible_idx.empty()) {
        update_passive_coeffs(coeffs, passive_ptr, passive_soln, npass);
        break;
      }

      extract_passive_coeffs(coeffs, passive_coeffs, passive_ptr, npass);

      const double alpha_max =
        compute_alpha_max(passive_coeffs, passive_soln, infeasible_idx);

      if (alpha_max <= tol) {

        update_coeffs_manual(coeffs, passive_soln, passive_ptr, updated_passive,
                             is_active, infeasible_idx, feasible_idx);
      } else {

        stepped_soln.noalias() = alpha_max * passive_soln;
        stepped_soln.noalias() += passive_coeffs * (1 - alpha_max);
        update_passive_coeffs(coeffs, passive_ptr, stepped_soln, npass);
        update_active_after_step(coeffs, passive, updated_passive, is_active,
                                 tol);
      }
      std::swap(passive, updated_passive);
    }
    neg_gradient = Xty;
    neg_gradient.noalias() -= XtX * coeffs;
  }
  return coeffs;
}
