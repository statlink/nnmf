#pragma once

#include "types.h"
#include <Eigen/Dense>

Vec fnnls_core(const Mat &XtX, const Vec &Xty, const double tol = 1e-5,
               const int max_iter = 1000);
