#pragma once

#include <cmath>

namespace cumlrkt::glm::logistic {

__device__ inline double softplus(double v) {
  return (v > 0 ? v : 0.0) + log1p(exp(-fabs(v)));
}

}  // namespace cumlrkt::glm::logistic
