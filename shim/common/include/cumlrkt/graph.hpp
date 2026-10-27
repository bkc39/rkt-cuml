#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <string>

#include "raftrkt/array.h"

namespace cumlrkt {

struct graph_views {
  const rr_view* rows;
  const rr_view* cols;
  const rr_view* vals;
};

struct coo {
  int32_t* rows;
  int32_t* cols;
  float* vals;
  int64_t nnz;
};

struct graph_scan {
  int64_t first_fault;
  bool any_positive;
  double total;
};

graph_scan scan_graph(const coo& c, int64_t n, rmm::cuda_stream_view stream);

coo checked_graph(const graph_views& g);

void require_samples(int64_t n, int64_t nnz, int64_t least);

std::string fault_message(const coo& c, int64_t i, int64_t n,
                          rmm::cuda_stream_view stream);

void require_sound_graph(const coo& c, int64_t n, rmm::cuda_stream_view stream);

}  // namespace cumlrkt
