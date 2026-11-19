#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

#include "ritz.hpp"

namespace cumlrkt::glm {

constexpr int64_t narrow_gram_columns = ritz::narrow_gram_columns;

template <typename T>
struct block_rows_of {
  const T* x;
  const T* w;
  const double* mean;
  const double* scale;
  int64_t n;
  int64_t d;
};

template <typename T>
void narrow_block_grams(rmm::cuda_stream_view stream, block_rows_of<T> rows,
                        int64_t first, int64_t count, double* parts);

}  // namespace cumlrkt::glm
