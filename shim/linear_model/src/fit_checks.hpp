#pragma once

#include <cstdint>
#include <functional>
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

template <typename T>
struct checked_fit {
  fit_inputs<T> in;
  double largest_weight;
};

struct fit_request {
  const rr_view* x;
  const rr_view* y;
  const rr_view* w;
  int32_t fit_intercept;
  int32_t algorithm;
};

int64_t workspace_of(glm_solver solver, ols_shape shape, int64_t lwork);

void require_workspace(glm_solver solver, ols_shape shape, int64_t lwork);

template <typename T>
int64_t cusolver_lwork_of(const raft::handle_t& handle, glm_solver solver,
                          ols_shape shape);

void require_fit_intercept(int32_t fit_intercept);

void require_ols_codes(const fit_request& request);

template <typename T>
checked_fit<T> checked_inputs(const raft::handle_t& handle,
                              const fit_request& request,
                              const std::function<void()>& require_codes);

}  // namespace cumlrkt::glm
