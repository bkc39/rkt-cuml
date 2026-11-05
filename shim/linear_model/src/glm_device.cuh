#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <climits>
#include <cstdint>
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

struct column_of {
  int64_t n;

  __device__ int64_t operator()(int64_t k) const {
    return k / n;
  }
};

template <typename F, typename V>
struct by_column {
  F f;
  int64_t n;

  __device__ V operator()(int64_t k) const {
    const int64_t j = k / n;
    return f(k - (j * n), j);
  }
};

template <typename V, typename F, typename Op>
std::vector<V> column_totals(rmm::cuda_stream_view stream, int64_t n, int64_t d,
                             F f, Op op) {
  buffer<V> totals = scratch<V>(d, stream);
  buffer<int64_t> columns = scratch<int64_t>(d, stream);
  const thrust::counting_iterator<int64_t> k(0);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(k, column_of{n}),
      thrust::make_transform_iterator(k + (n * d), column_of{n}),
      thrust::make_transform_iterator(k, by_column<F, V>{f, n}), columns.data(),
      totals.data(), thrust::equal_to<int64_t>{}, op);
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

inline constexpr int no_exponent = INT_MIN;

// a·b·c as a significand product and the exponent ilogb(a)+ilogb(b)+ilogb(c),
// so a sum of such terms can be scaled by its largest term's power of two
// before any product underflows.
struct product3 {
  double significand;
  int exponent;
};

inline __device__ product3 product_of(double a, double b, double c) {
  if (a == 0 || b == 0 || c == 0) {
    return {0.0, no_exponent};
  }
  const int ea = ilogb(a);
  const int eb = ilogb(b);
  const int ec = ilogb(c);
  return {scalbn(a, -ea) * scalbn(b, -eb) * scalbn(c, -ec), ea + eb + ec};
}

inline __device__ double scaled_product(product3 p, int largest) {
  return p.exponent == no_exponent
             ? 0.0
             : scalbn(p.significand, p.exponent - largest);
}

struct largest_exponent {
  __device__ int operator()(int a, int b) const {
    return a < b ? b : a;
  }
};

}  // namespace cumlrkt::glm
