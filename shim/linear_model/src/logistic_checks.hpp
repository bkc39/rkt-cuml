#pragma once

#include <cmath>
#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::logistic {

template <typename T>
void require_finite_solution(const T* params, int64_t n,
                             rmm::cuda_stream_view stream) {
  if (first_non_finite(params, n, stream) >= 0) {
    throw logic_error("X: the qn solve gave a non-finite coefficient");
  }
}

template <typename T>
void require_coef_in_range(const T* coef, int64_t n,
                           rmm::cuda_stream_view stream) {
  if (first_non_finite(coef, n, stream) >= 0) {
    throw logic_error(std::format("X: a coefficient exceeds the {} range",
                                  raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
}

inline void require_fit_bounds(const fit_options& o) {
  if (!(o.C > 0) || !std::isfinite(o.C)) {
    throw logic_error(std::format("C: expected a positive finite real, got {}",
                                  printed(o.C)));
  }
  if (!(o.l1_ratio >= 0 && o.l1_ratio <= 1)) {
    throw logic_error(std::format(
        "l1-ratio: expected a real from 0 to 1, got {}", printed(o.l1_ratio)));
  }
  require_finite_non_negative("tol", o.tol);
  require_at_least("max-iter", o.max_iter, 1);
  require_at_least("linesearch-max-iter", o.linesearch_max_iter, 1);
  require_at_least("lbfgs-memory", o.lbfgs_memory, 1);
}

}  // namespace cumlrkt::glm::logistic
