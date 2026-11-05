#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
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
inline constexpr int row_lanes = 8;
inline constexpr int64_t chunk_rows = 2048;
inline constexpr int fold_threads = 256;
inline constexpr int64_t most_blocks = 65535;

// Column j's total over the selected rows depends only on its own values and
// rows.count: lane r of a chunk folds rows r, r + row_lanes, ... in order,
// the lanes fold in order, and fold_chunks folds the chunks in a fixed tree.
// So a zero-weight row compacted out and a deleted row give the same bits,
// and no other column, nor its position, can change them.
template <typename V, typename F, typename Op>
__global__ void chunk_totals(F f, Op op, V identity, row_selection rows,
                             int64_t d, V* partial) {
  __shared__ V lanes[row_lanes][column_lanes];
  const int lane = static_cast<int>(threadIdx.x) % column_lanes;
  const int r = static_cast<int>(threadIdx.x) / column_lanes;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk_rows;
  const int64_t end = std::min(begin + chunk_rows, rows.count);
  for (int64_t tile = blockIdx.y; tile * column_lanes < d; tile += gridDim.y) {
    const int64_t j = (tile * column_lanes) + lane;
    V acc = identity;
    if (j < d) {
#pragma unroll 4
      for (int64_t p = begin + r; p < end; p += row_lanes) {
        acc = op(acc, f(source_row(rows, p), j));
      }
    }
    lanes[r][lane] = acc;
    __syncthreads();
    if (r == 0 && j < d) {
      for (int k = 1; k < row_lanes; ++k) {
        acc = op(acc, lanes[k][lane]);
      }
      partial[(static_cast<int64_t>(blockIdx.x) * d) + j] = acc;
    }
    __syncthreads();
  }
}

template <typename V, typename Op>
__global__ void fold_chunks(Op op, V identity, int64_t chunks, int64_t d,
                            const V* partial, V* totals) {
  __shared__ V folded[fold_threads];
  const int t = static_cast<int>(threadIdx.x);
  for (int64_t j = blockIdx.x; j < d; j += gridDim.x) {
    V acc = identity;
    for (int64_t b = t; b < chunks; b += fold_threads) {
      acc = op(acc, partial[(b * d) + j]);
    }
    folded[t] = acc;
    __syncthreads();
    for (int s = fold_threads / 2; s > 0; s /= 2) {
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

// One total per column j of f(i, j) over the selected source rows i. A warp
// reads column_lanes adjacent columns of one row of the row-major X.
template <typename V, typename F, typename Op>
std::vector<V> column_reduce(rmm::cuda_stream_view stream, row_selection rows,
                             int64_t d, F f, Op op, V identity) {
  if (d == 0) {
    return {};
  }
  const int64_t chunks =
      std::max<int64_t>(1, (rows.count + chunk_rows - 1) / chunk_rows);
  const int64_t tiles = (d + column_lanes - 1) / column_lanes;
  buffer<V> partial = scratch<V>(chunks * d, stream);
  buffer<V> totals = scratch<V>(d, stream);
  const dim3 grid(static_cast<unsigned>(chunks),
                  static_cast<unsigned>(std::min(tiles, most_blocks)));
  chunk_totals<V><<<grid, column_lanes * row_lanes, 0, stream.value()>>>(
      f, op, identity, rows, d, partial.data());
  raftrkt::cuda_check(cudaGetLastError(), "chunk_totals");
  fold_chunks<V><<<static_cast<unsigned>(std::min(d, most_blocks)),
                   fold_threads, 0, stream.value()>>>(
      op, identity, chunks, d, partial.data(), totals.data());
  raftrkt::cuda_check(cudaGetLastError(), "fold_chunks");
  return to_host(totals, stream);
}

struct larger {
  __device__ double operator()(double a, double b) const {
    return a < b ? b : a;
  }
};

template <typename T>
double total(rmm::cuda_stream_view stream, int64_t n, T f) {
  const thrust::counting_iterator<int64_t> i(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), i, i + n, f, 0.0,
                                  thrust::plus<double>{});
}

template <typename T>
struct weighted_value {
  const T* y;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i) * static_cast<double>(y[i]);
  }
};

template <typename T>
struct weight_of {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i);
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

inline constexpr int vector_threads = 256;

// The total of f(i) over the selected rows i, in an order fixed by rows.count:
// thread t of a chunk folds its rows t, t + vector_threads, ... in order, the
// threads fold in a fixed tree, and fold_chunks folds the chunks.
template <typename V, typename F, typename Op>
__global__ void vector_chunks(F f, Op op, V identity, row_selection rows,
                              V* partial) {
  __shared__ V folded[vector_threads];
  const int t = static_cast<int>(threadIdx.x);
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk_rows;
  const int64_t end = std::min(begin + chunk_rows, rows.count);
  V acc = identity;
  for (int64_t p = begin + t; p < end; p += vector_threads) {
    acc = op(acc, f(source_row(rows, p)));
  }
  folded[t] = acc;
  __syncthreads();
  for (int s = vector_threads / 2; s > 0; s /= 2) {
    if (t < s) {
      folded[t] = op(folded[t], folded[t + s]);
    }
    __syncthreads();
  }
  if (t == 0) {
    partial[blockIdx.x] = folded[0];
  }
}

template <typename V, typename F, typename Op>
V vector_reduce(rmm::cuda_stream_view stream, row_selection rows, F f, Op op,
                V identity) {
  const int64_t chunks =
      std::max<int64_t>(1, (rows.count + chunk_rows - 1) / chunk_rows);
  buffer<V> partial = scratch<V>(chunks, stream);
  buffer<V> totals = scratch<V>(1, stream);
  vector_chunks<V>
      <<<static_cast<unsigned>(chunks), vector_threads, 0, stream.value()>>>(
          f, op, identity, rows, partial.data());
  raftrkt::cuda_check(cudaGetLastError(), "vector_chunks");
  fold_chunks<V><<<1, fold_threads, 0, stream.value()>>>(
      op, identity, chunks, 1, partial.data(), totals.data());
  raftrkt::cuda_check(cudaGetLastError(), "fold_chunks");
  return to_host(totals, stream)[0];
}

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

// 2^exponent as a T factor when it is one, else 0. scaled_by(v, p) is then
// scaled(v, p.exponent): v·2^e rounds once either way.
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

// Tested on the exponent bits, so it costs no double-rate instruction.
inline __device__ bool moderate(double v) {
  const int biased = (__double2hiint(v) >> 20) & 0x7ff;
  return biased >= 1023 - moderate_reach && biased <= 1023 + moderate_reach;
}

inline __host__ __device__ product3 no_product() {
  return {0.0, no_exponent};
}

inline __device__ product3 product_of(double a, double b) {
  if (moderate(a) && moderate(b)) {
    return {a * b, 0};
  }
  if (a == 0 || b == 0) {
    return no_product();
  }
  const int ea = ilogb(a);
  const int eb = ilogb(b);
  return {scalbn(a, -ea) * scalbn(b, -eb), ea + eb};
}

inline __device__ product3 product_of(double a, double b, double c) {
  if (moderate(a) && moderate(b) && moderate(c)) {
    return {a * b * c, 0};
  }
  if (a == 0 || b == 0 || c == 0) {
    return no_product();
  }
  const int ea = ilogb(a);
  const int eb = ilogb(b);
  const int ec = ilogb(c);
  return {scalbn(a, -ea) * scalbn(b, -eb) * scalbn(c, -ec), ea + eb + ec};
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
