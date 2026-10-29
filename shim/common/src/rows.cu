#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "cumlrkt/rows.hpp"

namespace cumlrkt {

namespace {

template <typename T>
struct squared_norm {
  const T* x;
  int64_t d;

  __device__ double operator()(int64_t i) const {
    const T* row = x + i * d;
    double sum{0};
    for (int64_t j = 0; j < d; ++j) {
      sum += static_cast<double>(row[j]) * static_cast<double>(row[j]);
    }
    return sum;
  }
};

struct past {
  double most;

  __device__ bool operator()(double s) const {
    return !(s <= most);
  }
};

template <typename T>
struct zero_row {
  const T* x;
  int64_t d;

  __device__ bool operator()(int64_t i) const {
    const T* row = x + i * d;
    for (int64_t j = 0; j < d; ++j) {
      if (row[j] != 0) {
        return false;
      }
    }
    return true;
  }
};

template <typename T>
struct non_finite {
  const T* x;

  __device__ bool operator()(int64_t i) const {
    return !isfinite(x[i]);
  }
};

template <typename T>
struct large {
  const T* x;
  double most;

  __device__ bool operator()(int64_t i) const {
    return !(fabs(static_cast<double>(x[i])) <= most);
  }
};

template <typename T>
struct magnitude {
  __device__ double operator()(T v) const {
    return fabs(static_cast<double>(v));
  }
};

struct larger {
  __device__ double operator()(double a, double b) const {
    return isnan(a) || a > b ? a : b;
  }
};

template <typename T>
struct unlike_first {
  const T* x;
  int64_t d;

  __device__ bool operator()(int64_t i) const {
    const T* row = x + i * d;
    for (int64_t j = 0; j < d; ++j) {
      if (row[j] != x[j]) {
        return true;
      }
    }
    return false;
  }
};

template <typename Predicate>
int64_t first_index(int64_t start, int64_t end, Predicate p,
                    rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> indices(start);
  const int64_t i = *thrust::find_if(rmm::exec_policy(stream), indices,
                                     indices + (end - start), p);
  return i == end ? -1 : i;
}

}  // namespace

template <typename T>
std::pair<int64_t, double> first_large_row(const T* x, int64_t n, int64_t d,
                                           double most,
                                           rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const squared_norm<T> norm{x, d};
  const auto norms = thrust::make_transform_iterator(rows, norm);
  const int64_t i =
      thrust::find_if(rmm::exec_policy(stream), norms, norms + n, past{most}) -
      norms;
  if (i == n) {
    return {-1, 0};
  }
  const double s =
      thrust::transform_reduce(rmm::exec_policy(stream), rows + i, rows + i + 1,
                               norm, 0.0, thrust::plus<double>{});
  return {i, s};
}

template <typename T>
int64_t first_zero_row(const T* x, int64_t n, int64_t d,
                       rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const int64_t i = *thrust::find_if(rmm::exec_policy(stream), rows, rows + n,
                                     zero_row<T>{x, d});
  return i == n ? -1 : i;
}

template <typename T>
int64_t first_non_finite(const T* x, int64_t count,
                         rmm::cuda_stream_view stream) {
  return first_index(0, count, non_finite<T>{x}, stream);
}

template <typename T>
int64_t first_large_element(const T* x, int64_t count, double most,
                            rmm::cuda_stream_view stream) {
  return first_index(0, count, large<T>{x, most}, stream);
}

template <typename T>
double largest_magnitude(const T* x, int64_t count,
                         rmm::cuda_stream_view stream) {
  return thrust::transform_reduce(rmm::exec_policy(stream), x, x + count,
                                  magnitude<T>{}, 0.0, larger{});
}

template <typename T>
int64_t first_row_unlike_first(const T* x, int64_t n, int64_t d,
                               rmm::cuda_stream_view stream) {
  return first_index(1, n, unlike_first<T>{x, d}, stream);
}

template std::pair<int64_t, double> first_large_row<float>(
    const float*, int64_t, int64_t, double, rmm::cuda_stream_view);
template std::pair<int64_t, double> first_large_row<double>(
    const double*, int64_t, int64_t, double, rmm::cuda_stream_view);
template int64_t first_zero_row<float>(const float*, int64_t, int64_t,
                                       rmm::cuda_stream_view);
template int64_t first_zero_row<double>(const double*, int64_t, int64_t,
                                        rmm::cuda_stream_view);
template int64_t first_non_finite<float>(const float*, int64_t,
                                         rmm::cuda_stream_view);
template int64_t first_non_finite<double>(const double*, int64_t,
                                          rmm::cuda_stream_view);
template int64_t first_large_element<float>(const float*, int64_t, double,
                                            rmm::cuda_stream_view);
template int64_t first_large_element<double>(const double*, int64_t, double,
                                             rmm::cuda_stream_view);
template double largest_magnitude<float>(const float*, int64_t,
                                         rmm::cuda_stream_view);
template double largest_magnitude<double>(const double*, int64_t,
                                          rmm::cuda_stream_view);
template int64_t first_row_unlike_first<float>(const float*, int64_t, int64_t,
                                               rmm::cuda_stream_view);
template int64_t first_row_unlike_first<double>(const double*, int64_t, int64_t,
                                                rmm::cuda_stream_view);

}  // namespace cumlrkt
