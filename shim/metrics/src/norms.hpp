#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

namespace cumlrkt::metrics {

template <typename T>
std::pair<int64_t, T> first_normless_row(const T* x, int64_t n, int64_t d,
                                         rmm::cuda_stream_view stream);

}  // namespace cumlrkt::metrics
