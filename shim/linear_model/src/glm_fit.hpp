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
  glm_solver solver;
  counted_rows counted;
};

template <typename T>
struct prepared_fit {
  buffer<T> weights;
  fit_inputs<T> in;
  bool fit_intercept;
  int weight_power;
  int64_t positive;
  counted_rows counted;
  staging planned;
};

template <typename T>
prepared_fit<T> prepare_fit(const raft::handle_t& handle,
                            const checked_fit<T>& checked,
                            int32_t fit_intercept);

template <typename T>
double finish_fit(const raft::handle_t& handle, const fit_inputs<T>& in,
                  const fit_plan& plan, T alpha, rr_deposit* out);

template <typename T>
double ols_fit_prepared(const raft::handle_t& handle, prepared_fit<T> prepared,
                        int algorithm, rr_deposit* out);

template <typename T>
double ols_fit_into(const raft::handle_t& handle, const fit_request& request,
                    rr_deposit* out);

}  // namespace cumlrkt::glm
