#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

struct cd_centres {
  std::vector<double> mean;
  std::vector<double> remainder;
  centre y;
};

template <typename T>
std::optional<cd_centres> centres_of(const T* y, const T* w, row_selection rows,
                                     const staging& s, bool fit_intercept,
                                     rmm::cuda_stream_view stream);

template <typename T>
double weighted_intercept(const cd_centres& c, const std::vector<T>& b,
                          int y_exponent) {
  double fitted = 0;
  for (size_t k = 0; k < b.size(); ++k) {
    fitted += (c.mean[k] + c.remainder[k]) * static_cast<double>(b[k]);
  }
  return std::ldexp((c.y.mean + c.y.remainder) - fitted, -y_exponent);
}

template <typename T>
staged_problem<T> stage_centred(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, row_selection rows,
                                const staging& s, const cd_centres& c);

template <typename T>
double weight_total(const T* w, row_selection rows,
                    rmm::cuda_stream_view stream);

template <typename T>
staged_problem<T> stage_weighted(const raft::handle_t& handle,
                                 const matrix<T, int64_t>& x, const T* y,
                                 const T* w, row_selection rows,
                                 const staging& s,
                                 const std::optional<cd_centres>& c,
                                 double total);

}  // namespace cumlrkt::glm
