#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::metrics {

template <typename T>
int64_t first_normless_row(const T* x, int64_t n, int64_t d,
                           rmm::cuda_stream_view stream);

}  // namespace cumlrkt::metrics
