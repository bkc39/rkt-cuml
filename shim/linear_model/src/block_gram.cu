#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

#include "block_gram.hpp"
#include "cumlrkt/support.hpp"
#include "ritz.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int tile_rows = 256;
constexpr int64_t block_rows = ritz::gram_block_rows;

template <typename T>
__global__ void narrow_grams(block_rows_of<T> r, int64_t first, float* parts) {
  __shared__ float tile[tile_rows * narrow_gram_columns];
  const int64_t d = r.d;
  const int64_t cells = d * d;
  const int t = static_cast<int>(threadIdx.x);
  const int64_t column = t / d;
  const int64_t row = t - (column * d);
  const int64_t begin = (first + blockIdx.x) * block_rows;
  float s = 0.0F;
  for (int64_t start = begin; start < begin + block_rows; start += tile_rows) {
    const int64_t i = start + t;
    for (int64_t k = 0; k < d; ++k) {
      float v = 0.0F;
      if (i < r.n) {
        const double root =
            r.w == nullptr ? 1.0 : sqrt(static_cast<double>(r.w[i]));
        v = static_cast<float>(
            root * r.scale[k] *
            (static_cast<double>(r.x[(k * r.n) + i]) - r.mean[k]));
      }
      tile[(t * d) + k] = v;
    }
    __syncthreads();
    if (t < cells) {
      for (int p = 0; p < tile_rows; ++p) {
        s = fmaf(tile[(p * d) + row], tile[(p * d) + column], s);
      }
    }
    __syncthreads();
  }
  if (t < cells) {
    parts[(blockIdx.x * cells) + t] = s;
  }
}

}  // namespace

template <typename T>
void narrow_block_grams(rmm::cuda_stream_view stream, block_rows_of<T> rows,
                        int64_t first, int64_t count, float* parts) {
  narrow_grams<T>
      <<<static_cast<unsigned>(count), tile_rows, 0, stream.value()>>>(
          rows, first, parts);
  raftrkt::cuda_check(cudaGetLastError(), "narrow_grams");
}

template void narrow_block_grams<float>(rmm::cuda_stream_view,
                                        block_rows_of<float>, int64_t, int64_t,
                                        float*);
template void narrow_block_grams<double>(rmm::cuda_stream_view,
                                         block_rows_of<double>, int64_t,
                                         int64_t, float*);

}  // namespace cumlrkt::glm
