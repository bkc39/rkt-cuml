#pragma once

#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"

namespace cumlrkt::manifold {

inline void require_solved(const float* e, int64_t n, int64_t k,
                    rmm::cuda_stream_view stream) {
  const int64_t i = first_non_finite(e, n * k, stream);
  if (i >= 0) {
    const float got = read_one(e, i, stream);
    throw logic_error(std::format(
        "embedding: row {}, column {}: the eigensolver returned {}", i / k,
        i % k, printed(static_cast<double>(got))));
  }
}

inline void require_nonzero_columns(const float* columns, int64_t n, int64_t k,
                             const char* advice, rmm::cuda_stream_view stream) {
  const int64_t c = first_zero_row(columns, k, n, stream);
  if (c >= 0) {
    throw logic_error(
        std::format("embedding: column {} is zero: the eigensolver found no "
                    "eigenvector of this graph{}",
                    c, advice));
  }
}

}  // namespace cumlrkt::manifold
