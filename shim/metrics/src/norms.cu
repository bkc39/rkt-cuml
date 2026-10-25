#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "norms.hpp"

namespace cumlrkt::metrics {

namespace {

template <typename T>
struct squared_norm {
  const T* x;
  int64_t d;

  __device__ T operator()(int64_t i) const {
    const T* row = x + i * d;
    T sum{0};
    for (int64_t j = 0; j < d; ++j) {
      sum += row[j] * row[j];
    }
    return sum;
  }
};

template <typename T>
struct below {
  T least;

  __device__ bool operator()(T sum) const {
    return !(sum >= least);
  }
};

}  // namespace

template <typename T>
std::pair<int64_t, T> first_normless_row(const T* x, int64_t n, int64_t d,
                                         T least,
                                         rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const squared_norm<T> norm{x, d};
  const auto norms = thrust::make_transform_iterator(rows, norm);
  const int64_t i = thrust::find_if(rmm::exec_policy(stream), norms, norms + n,
                                    below<T>{least}) -
                    norms;
  if (i == n) {
    return {-1, T{0}};
  }
  const T s =
      thrust::transform_reduce(rmm::exec_policy(stream), rows + i, rows + i + 1,
                               norm, T{0}, thrust::plus<T>{});
  return {i, s};
}

template std::pair<int64_t, float> first_normless_row<float>(
    const float*, int64_t, int64_t, float, rmm::cuda_stream_view);
template std::pair<int64_t, double> first_normless_row<double>(
    const double*, int64_t, int64_t, double, rmm::cuda_stream_view);

}  // namespace cumlrkt::metrics
