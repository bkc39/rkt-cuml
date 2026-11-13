#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <string>
#include <vector>

#include "cumlrkt/support.hpp"
#include "logistic_device.hpp"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"

namespace cumlrkt::glm::qn {

inline constexpr int quiet_level = 6;

struct qn_request {
  const rr_view* x;
  const rr_view* y;
  const rr_view* w;
  const rr_view* keys;
  const rr_view* values;
  int32_t penalty;
  double C;
  double l1_ratio;
  double tol;
  int32_t max_iter;
  int32_t linesearch_max_iter;
  int32_t lbfgs_memory;
  int32_t fit_intercept;
  int32_t normalized;
  int32_t class_mode;
};

struct qn_check {
  qn_shape shape;
  std::vector<double> sigma;
  std::vector<double> mean;
  double pen_l1;
  double pen_l2;
  int n_iter;
};

void require_qn_codes(const qn_request& r);

int32_t logistic_fit(const raft::handle_t& handle, const qn_request& r,
                     rr_deposit* out);

template <typename Fn>
void on_class_labels(const rr_view* y, Fn&& fn) {
  const rr_view& view = *raftrkt::require(y, "y");
  switch (view.dtype) {
    case RR_DTYPE_INT32:
      fn(int32_t{});
      break;
    case RR_DTYPE_INT64:
      fn(int64_t{});
      break;
    case RR_DTYPE_FLOAT32:
      fn(float{});
      break;
    case RR_DTYPE_FLOAT64:
      fn(double{});
      break;
    default:
      throw logic_error(
          std::string("y: expected int32, int64, float32 or float64, got ") +
          raftrkt::dtype_name(view.dtype));
  }
}

}  // namespace cumlrkt::glm::qn
