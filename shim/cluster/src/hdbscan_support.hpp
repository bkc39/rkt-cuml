#pragma once

#include <cstdint>
#include <format>
#include <limits>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/common.hpp"
#include "linkage_device.hpp"
#include "support.hpp"

namespace cumlrkt::cluster::hdbscan_limits {

inline constexpr int64_t most_index = std::numeric_limits<int32_t>::max();
inline constexpr double most_squared_norm = 0x1p125;

inline void require_bounded_rows(const char* name,
                                 const matrix<float, int64_t>& x,
                                 rmm::cuda_stream_view stream) {
  const auto [row, squared] =
      first_large_row(x.data, x.n, x.d, most_squared_norm, stream);
  if (row >= 0) {
    throw logic_error(
        std::format("{}: row {}: expected a squared norm of at most {}, got {}",
                    name, row, printed(most_squared_norm), printed(squared)));
  }
}

inline void require_index(const char* name, int64_t rows, int64_t cols) {
  if (cols > 0 && rows > most_index / cols) {
    throw logic_error(std::format(
        "{}: {}x{} elements do not fit the index type", name, rows, cols));
  }
}

}  // namespace cumlrkt::cluster::hdbscan_limits
