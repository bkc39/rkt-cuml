#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

namespace cumlrkt::cluster {

std::pair<int64_t, double> first_large_row(const float* x, int64_t n, int64_t d,
                                           double most,
                                           rmm::cuda_stream_view stream);

template <typename Index>
void ascending_pairs(Index* children, int64_t rows,
                     rmm::cuda_stream_view stream);

}  // namespace cumlrkt::cluster
