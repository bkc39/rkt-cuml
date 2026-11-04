#pragma once

#include <raft/core/handle.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

struct scaled_sum {
  double value;
  int exponent;
};

struct moment {
  scaled_sum square;
  scaled_sum cross;
};

struct residual_terms {
  std::vector<moment> moments;
  std::vector<double> largest;
  scaled_sum y_square;
};

template <typename T>
residual_terms residual_moments(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, staged_columns columns,
                                const T* coef, bool fit_intercept);

}  // namespace cumlrkt::glm
