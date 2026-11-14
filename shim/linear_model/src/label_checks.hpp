#pragma once

#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>

#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "logistic_device.hpp"

namespace cumlrkt::glm {

template <typename L>
void require_integral(const L* y, int64_t n, rmm::cuda_stream_view stream) {
  if constexpr (!std::is_integral_v<L>) {
    const int64_t i = logistic::first_non_integral(y, n, stream);
    if (i >= 0) {
      throw logic_error(std::format("y: expected integral class labels, got {}",
                                    printed(to_host(y + i, 1, stream)[0])));
    }
  }
}

}  // namespace cumlrkt::glm
