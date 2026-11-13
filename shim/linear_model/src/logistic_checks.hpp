#pragma once

#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
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

}  // namespace cumlrkt::glm::logistic
