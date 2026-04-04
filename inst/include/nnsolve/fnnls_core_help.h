#pragma once

#include "types.h"
#include <Eigen/Dense>
#include <vector>


bool add_kkt_violators(const Vec &neg_gradient, std::vector<char> &is_active,
                       std::vector<int> &passive, const int ncoef,
                       const double tol);

void grab_passive_idx(const Mat &XtX, const Vec &Xty, Mat &XtX_pass,
                      Vec &Xty_pass, const int *passive_ptr, const int npass,
                      const int ncoef);

void update_passive_coeffs(Vec &coeffs, const int *passive_ptr,
                           const Vec &passive_soln, const int npass);

void extract_passive_coeffs(const Vec &coeffs, Vec &pass_coeffs,
                            const int *passive_ptr, const int npass);

void split_by_feasibility(const Vec &passive_soln,
                          std::vector<int> &feasible_idx,
                          std::vector<int> &infeasible_idx, const double tol,
                          const int npass);

double compute_alpha_max(const Vec &passive_coeffs, const Vec &passive_soln,
                         const std::vector<int> &infeasible_idx);

void update_coeffs_manual(Vec &coeffs, const Vec &passive_soln,
                          const int *passive_ptr,
                          std::vector<int> &updated_passive,
                          std::vector<char> &is_active,
                          const std::vector<int> &infeasible_idx,
                          const std::vector<int> &feasible_idx);

void update_active_after_step(Vec &coeffs, const std::vector<int> &passive,
                              std::vector<int> &updated_passive,
                              std::vector<char> &is_active, const double tol);

