#include <cublas_v2.h>

#include <cstdint>
#include <raft/core/cublas_macros.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>

#include "block_gram.hpp"
#include "cumlrkt/support.hpp"
#include "ritz.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int tile_rows = 256;
constexpr int run_rows = ritz::narrow_run_rows;
constexpr int64_t block_rows = ritz::gram_block_rows;
static_assert(tile_rows % run_rows == 0);
static_assert(block_rows % tile_rows == 0);

template <typename T>
__global__ void narrow_grams(block_rows_of<T> r, int64_t first, double* parts) {
  __shared__ float tile[tile_rows * narrow_gram_columns];
  const int64_t d = r.d;
  const int64_t cells = d * d;
  const int t = static_cast<int>(threadIdx.x);
  const int64_t column = t / d;
  const int64_t row = t - (column * d);
  const int64_t begin = (first + blockIdx.x) * block_rows;
  double total = 0.0;
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
      for (int run = 0; run < tile_rows; run += run_rows) {
        float s = 0.0F;
        for (int p = run; p < run + run_rows; ++p) {
          s = fmaf(tile[(p * d) + row], tile[(p * d) + column], s);
        }
        total += static_cast<double>(s);
      }
    }
    __syncthreads();
  }
  if (t < cells) {
    parts[(blockIdx.x * cells) + t] = total;
  }
}

class pedantic_math {
 public:
  explicit pedantic_math(cublasHandle_t cublas) : cublas_(cublas) {
    RAFT_CUBLAS_TRY(cublasGetMathMode(cublas_, &mode_));
    RAFT_CUBLAS_TRY(cublasSetMathMode(cublas_, CUBLAS_PEDANTIC_MATH));
  }
  pedantic_math(const pedantic_math&) = delete;
  pedantic_math& operator=(const pedantic_math&) = delete;
  pedantic_math(pedantic_math&&) = delete;
  pedantic_math& operator=(pedantic_math&&) = delete;
  ~pedantic_math() {
    cublasSetMathMode(cublas_, mode_);
  }

 private:
  cublasHandle_t cublas_;
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
};

}  // namespace

template <typename U>
void wide_block_grams(cublasHandle_t cublas, const U* block, int64_t rows,
                      int64_t d, ritz::gram_batch batch, U* parts) {
  const pedantic_math pedantic(cublas);
  const U one{1};
  const U zero{0};
  const auto dim = static_cast<int>(d);
  const auto run = static_cast<int>(batch.run);
  const auto lead = static_cast<int>(rows);
  if constexpr (std::is_same_v<U, float>) {
    RAFT_CUBLAS_TRY(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, run, &one, block, lead,
        batch.run, block, lead, batch.run, &zero, parts, dim, d * d,
        static_cast<int>(batch.blocks)));
  } else {
    RAFT_CUBLAS_TRY(cublasDgemmStridedBatched(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, run, &one, block, lead,
        batch.run, block, lead, batch.run, &zero, parts, dim, d * d,
        static_cast<int>(batch.blocks)));
  }
}

template void wide_block_grams<float>(cublasHandle_t, const float*, int64_t,
                                      int64_t, ritz::gram_batch, float*);
template void wide_block_grams<double>(cublasHandle_t, const double*, int64_t,
                                       int64_t, ritz::gram_batch, double*);

template <typename T>
void narrow_block_grams(rmm::cuda_stream_view stream, block_rows_of<T> rows,
                        int64_t first, int64_t count, double* parts) {
  narrow_grams<T>
      <<<static_cast<unsigned>(count), tile_rows, 0, stream.value()>>>(
          rows, first, parts);
  raftrkt::cuda_check(cudaGetLastError(), "narrow_grams");
}

template void narrow_block_grams<float>(rmm::cuda_stream_view,
                                        block_rows_of<float>, int64_t, int64_t,
                                        double*);
template void narrow_block_grams<double>(rmm::cuda_stream_view,
                                         block_rows_of<double>, int64_t,
                                         int64_t, double*);

}  // namespace cumlrkt::glm
