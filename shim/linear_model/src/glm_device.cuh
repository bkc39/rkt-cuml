#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

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
