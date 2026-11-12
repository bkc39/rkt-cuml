#pragma once

#include <thrust/functional.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

template <typename T>
__device__ bool counted(const T* w, int64_t i) {
  return w == nullptr || w[i] > 0;
}

template <typename T>
__device__ double weight(const T* w, int64_t i) {
  return w == nullptr ? 1.0 : static_cast<double>(w[i]);
}

__device__ inline int64_t source_row(row_selection rows, int64_t p) {
  return rows.index == nullptr ? p : rows.index[p];
}

inline constexpr int column_lanes = 32;
inline constexpr int block_threads = 256;
inline constexpr int64_t chunk_rows = 2048;
inline constexpr int64_t most_blocks = 65535;

// Column j's total over the selected rows depends only on its own values and
// rows.count: row lane r of a chunk folds rows r, r + Rows, ... in order, the
// row lanes fold in a fixed tree, and fold_chunks folds the chunks in a fixed
// tree. So a zero-weight row compacted out and a deleted row give the same
// bits, and no other column, nor its position, can change them. Columns is
// fixed by the caller, never chosen from d.
template <int Columns, typename V, typename F, typename Op>
__global__ void chunk_totals(F f, Op op, V identity, row_selection rows,
                             int64_t d, V* partial) {
  constexpr int Rows = block_threads / Columns;
  __shared__ V lanes[block_threads];
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t % Columns;
  const int r = t / Columns;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk_rows;
  const int64_t end = std::min(begin + chunk_rows, rows.count);
  for (int64_t tile = blockIdx.y; tile * Columns < d; tile += gridDim.y) {
    const int64_t j = (tile * Columns) + lane;
    V acc = identity;
    if (j < d) {
#pragma unroll 4
      for (int64_t p = begin + r; p < end; p += Rows) {
        acc = op(acc, f(source_row(rows, p), j));
      }
    }
    lanes[t] = acc;
    __syncthreads();
    for (int s = Rows / 2; s > 0; s /= 2) {
      if (r < s) {
        lanes[t] = op(lanes[t], lanes[t + (s * Columns)]);
      }
      __syncthreads();
    }
    if (r == 0 && j < d) {
      partial[(static_cast<int64_t>(blockIdx.x) * d) + j] = lanes[lane];
    }
    __syncthreads();
  }
}

template <typename V, typename Op>
__global__ void fold_chunks(Op op, V identity, int64_t chunks, int64_t d,
                            const V* partial, V* totals) {
  __shared__ V folded[block_threads];
  const int t = static_cast<int>(threadIdx.x);
  for (int64_t j = blockIdx.x; j < d; j += gridDim.x) {
    V acc = identity;
    for (int64_t b = t; b < chunks; b += block_threads) {
      acc = op(acc, partial[(b * d) + j]);
    }
    folded[t] = acc;
    __syncthreads();
    for (int s = block_threads / 2; s > 0; s /= 2) {
      if (t < s) {
        folded[t] = op(folded[t], folded[t + s]);
      }
      __syncthreads();
    }
    if (t == 0) {
      totals[j] = folded[0];
    }
    __syncthreads();
  }
}

template <int Columns = column_lanes, typename V, typename F, typename Op>
std::vector<V> column_reduce(rmm::cuda_stream_view stream, row_selection rows,
                             int64_t d, F f, Op op, V identity) {
  if (d == 0) {
    return {};
  }
  const int64_t chunks =
      std::max<int64_t>(1, (rows.count + chunk_rows - 1) / chunk_rows);
  const int64_t tiles = (d + Columns - 1) / Columns;
  buffer<V> partial = scratch<V>(chunks * d, stream);
  buffer<V> totals = scratch<V>(d, stream);
  const dim3 grid(static_cast<unsigned>(chunks),
                  static_cast<unsigned>(std::min(tiles, most_blocks)));
  chunk_totals<Columns, V><<<grid, block_threads, 0, stream.value()>>>(
      f, op, identity, rows, d, partial.data());
  raftrkt::cuda_check(cudaGetLastError(), "chunk_totals");
  fold_chunks<V><<<static_cast<unsigned>(std::min(d, most_blocks)),
                   block_threads, 0, stream.value()>>>(
      op, identity, chunks, d, partial.data(), totals.data());
  raftrkt::cuda_check(cudaGetLastError(), "fold_chunks");
  return to_host(totals, stream);
}

template <typename F>
struct as_column {
  F f;

  __device__ auto operator()(int64_t i, int64_t /*j*/) const {
    return f(i);
  }
};

template <typename V, typename F, typename Op>
V vector_reduce(rmm::cuda_stream_view stream, row_selection rows, F f, Op op,
                V identity) {
  return column_reduce<1>(stream, rows, 1, as_column<F>{f}, op, identity)[0];
}

struct larger {
  __device__ double operator()(double a, double b) const {
    return a < b ? b : a;
  }
};

struct weighted_total {
  double weight;
  double value;
};

struct add_totals {
  __device__ weighted_total operator()(weighted_total a,
                                       weighted_total b) const {
    return {a.weight + b.weight, a.value + b.value};
  }
};

template <typename T>
struct row_weight {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return counted(w, i) ? weight(w, i) : 0.0;
  }
};

template <typename T>
double selected_weight(rmm::cuda_stream_view stream, row_selection rows,
                       const T* w) {
  return vector_reduce(stream, rows, row_weight<T>{w}, thrust::plus<double>{},
                       0.0);
}

template <typename T>
struct power_of_two {
  T factor;
  int exponent;
};

template <typename T>
power_of_two<T> power_of_two_of(int e) {
  constexpr int least =
      std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits;
  constexpr int most = std::numeric_limits<T>::max_exponent - 1;
  return {e >= least && e <= most ? std::ldexp(T{1}, e) : T{0}, e};
}

template <typename T>
__device__ T scaled_by(T v, power_of_two<T> p) {
  return p.factor != 0 ? cumlrkt::times(v, p.factor) : scaled(v, p.exponent);
}

template <typename T>
struct staged_target {
  const T* y;
  power_of_two<T> stage;

  __device__ double operator()(int64_t i) const {
    return static_cast<double>(scaled_by(y[i], stage));
  }
};

template <typename T>
struct weighted_target {
  staged_target<T> y;
  const T* w;

  __device__ weighted_total operator()(int64_t i) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    return {weight(w, i), weight(w, i) * y(i)};
  }
};

template <typename T>
struct target_deviation {
  staged_target<T> y;
  const T* w;
  double mean;

  __device__ double operator()(int64_t i) const {
    return counted(w, i) ? weight(w, i) * (y(i) - mean) : 0.0;
  }
};

inline constexpr int no_exponent = INT_MIN;

// significand·2^exponent. A product of factors in [2^-320, 2^321) is carried
// as itself at exponent 0, since neither it nor any partial product leaves
// double's normal range; any other as the product of the significands at
// the sum of the factors' ilogb. A sum of such terms is rescaled to the
// larger exponent at each addition, so no product underflows or overflows.
struct product3 {
  double significand;
  int exponent;
};

inline constexpr int moderate_reach = 320;

inline __device__ bool moderate(double v) {
  const int biased = (__double2hiint(v) >> 20) & 0x7ff;
  return biased >= 1023 - moderate_reach && biased <= 1023 + moderate_reach;
}

inline __host__ __device__ product3 no_product() {
  return {0.0, no_exponent};
}

template <typename... D>
__device__ product3 product_of(D... factor) {
  if ((moderate(factor) && ...)) {
    return {(... * factor), 0};
  }
  if (((factor == 0) || ...)) {
    return no_product();
  }
  return {(... * scalbn(factor, -ilogb(factor))), (... + ilogb(factor))};
}

struct add_scaled {
  __device__ product3 operator()(product3 a, product3 b) const {
    if (a.exponent == b.exponent) {
      return {a.significand + b.significand, a.exponent};
    }
    if (a.exponent == no_exponent) {
      return b;
    }
    if (b.exponent == no_exponent) {
      return a;
    }
    const int e = a.exponent < b.exponent ? b.exponent : a.exponent;
    return {scalbn(a.significand, a.exponent - e) +
                scalbn(b.significand, b.exponent - e),
            e};
  }
};

}  // namespace cumlrkt::glm
