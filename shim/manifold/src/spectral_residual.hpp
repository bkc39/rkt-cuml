#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/graph.hpp"

namespace cumlrkt::manifold {

struct column_fit {
  double residual;
  double spread;
};

std::vector<column_fit> column_fits(const raft::handle_t& handle, const coo& c,
                                    int64_t n, const float* columns, int64_t k,
                                    bool normalised);

double weight_span(const coo& c, rmm::cuda_stream_view stream);

}  // namespace cumlrkt::manifold
