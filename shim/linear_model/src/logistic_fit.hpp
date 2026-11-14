#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rapids_logger/logger.hpp>

#include "raftrkt/array.h"
#include "raftrkt/deposit.h"

namespace cumlrkt::glm::logistic {

inline constexpr int quiet_level =
    static_cast<int>(rapids_logger::level_enum::off);

struct fit_views {
  const rr_view* x;
  const rr_view* y;
  const rr_view* sample_weight;
  const rr_view* class_weight;
};

struct fit_codes {
  int32_t penalty;
  int32_t fit_intercept;
  int32_t normalized;
  int32_t class_mode;
};

struct fit_options {
  int32_t penalty;
  double C;
  double l1_ratio;
  double tol;
  int32_t max_iter;
  int32_t linesearch_max_iter;
  int32_t lbfgs_memory;
  bool fit_intercept;
  bool normalized;
  int32_t class_mode;
};

void require_fit_codes(const fit_codes& codes);

int32_t fit(const raft::handle_t& handle, const fit_views& views,
            const fit_options& options, rr_deposit* out);

}  // namespace cumlrkt::glm::logistic
