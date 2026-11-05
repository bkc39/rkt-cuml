#pragma once

#include <cstdint>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "raftrkt/deposit.h"

namespace cumlrkt::glm {

template <typename T>
struct fit_result {
  std::unique_ptr<buffer<T>> coef;
  double intercept;
};

struct counted_rows {
  buffer<int64_t> index;
  row_selection rows;
};

struct fit_plan {
  staging s;
  bool fit_intercept;
  int algorithm;
  counted_rows counted;
};

template <typename T>
counted_rows counted_rows_of(const T* w, int64_t n, int64_t positive,
                             rmm::cuda_stream_view stream);

template <typename T>
fit_result<T> nothing_to_solve(const raft::handle_t& handle,
                               const fit_inputs<T>& in, const fit_plan& plan,
                               bool constant_y);

template <typename T>
double deposit_fit(const raft::handle_t& handle, fit_result<T> r,
                   rr_deposit* out);

template <typename T>
double ols_fit_into(const raft::handle_t& handle, const fit_request& request,
                    rr_deposit* out);

}  // namespace cumlrkt::glm
