#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::metrics {

void count_pairs(int64_t* counts, const int32_t* a, const int32_t* b, int64_t n,
                 int64_t k1, int64_t k2, rmm::cuda_stream_view stream);

}  // namespace cumlrkt::metrics
