#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "raftrkt/array.h"

namespace cumlrkt::glm {

template <typename T>
struct fit_inputs {
  matrix<T, int64_t> x;
  const T* y;
  const T* w;
};

struct fit_request {
  const rr_view* x;
  const rr_view* y;
  const rr_view* w;
  int32_t fit_intercept;
  int32_t algorithm;
};

void require_workspace(int algorithm, int64_t n, int64_t d, int64_t lwork);

template <typename T>
int64_t cusolver_lwork_of(const raft::handle_t& handle, int algorithm,
                          ols_shape shape);

template <typename T>
fit_inputs<T> checked_inputs(const raft::handle_t& handle,
                             const fit_request& request);

}  // namespace cumlrkt::glm
