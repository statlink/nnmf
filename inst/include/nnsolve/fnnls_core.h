#pragma once

#include "types.h"
#include <Eigen/Dense>


Vec fnnls_core(const Mat &XtX, const Vec &Xty, const double tol,
               const int max_iter);

