#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::cluster {

struct graph_scan {
  int64_t first_fault;
  bool any_positive;
  double total;
};

graph_scan scan_graph(const int32_t* rows, const int32_t* cols,
                      const float* vals, int64_t nnz, int64_t n,
                      rmm::cuda_stream_view stream);

}  // namespace cumlrkt::cluster
