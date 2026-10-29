#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
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

struct refusal {
  std::size_t column;
  bool constant;
};

std::optional<refusal> first_refused(const std::vector<column_fit>& fits,
                                     bool drop_first);

struct graph_shape {
  bool symmetric;
  double span;
};

graph_shape shape_of(const raft::handle_t& handle, const coo& c, int64_t n);

}  // namespace cumlrkt::manifold
