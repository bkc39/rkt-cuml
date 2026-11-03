#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/support.hpp"
#include "raftrkt/array.h"

namespace cumlrkt::glm {

template <typename T>
struct fit_inputs {
  matrix<T, int64_t> x;
  const T* y;
  const T* w;
};

void require_workspace(int algorithm, int64_t n, int64_t d, int64_t lwork);

template <typename T>
int64_t cusolver_lwork_of(const raft::handle_t& handle, int algorithm,
                          int64_t n, int64_t d);

void require_codes(int32_t fit_intercept, int32_t algorithm);

template <typename T>
fit_inputs<T> checked_inputs(const raft::handle_t& handle,
                             const rr_view* x_view, const rr_view* y_view,
                             const rr_view* w_view, int32_t fit_intercept,
                             int32_t algorithm);

}  // namespace cumlrkt::glm
