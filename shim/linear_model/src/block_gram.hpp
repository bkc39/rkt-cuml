#pragma once

#include <cublas_v2.h>

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

template <typename U>
void wide_block_grams(cublasHandle_t cublas, const U* block, int64_t rows,
                      int64_t d, ritz::gram_batch batch, U* parts);

}  // namespace cumlrkt::glm
