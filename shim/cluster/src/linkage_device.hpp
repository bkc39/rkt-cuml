#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::cluster {

template <typename Index>
void ascending_pairs(Index* children, int64_t rows,
                     rmm::cuda_stream_view stream);

}  // namespace cumlrkt::cluster
