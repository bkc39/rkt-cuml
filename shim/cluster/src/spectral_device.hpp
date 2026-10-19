#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::cluster {

enum class graph_fault_kind : std::uint8_t {
  none = 0,
  row_index = 1,
  col_index = 2,
  self_loop = 3,
  weight = 4
};

struct graph_fault {
  int64_t entry;
  graph_fault_kind kind;
  double value;
  bool any_positive;
};

graph_fault first_graph_fault(const int32_t* rows, const int32_t* cols,
                              const float* vals, int64_t nnz, int64_t n,
                              rmm::cuda_stream_view stream);

}  // namespace cumlrkt::cluster
