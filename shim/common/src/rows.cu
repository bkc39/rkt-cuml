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

template std::pair<int64_t, double> first_large_row<float>(
    const float*, int64_t, int64_t, double, rmm::cuda_stream_view);
template std::pair<int64_t, double> first_large_row<double>(
    const double*, int64_t, int64_t, double, rmm::cuda_stream_view);
template int64_t first_zero_row<float>(const float*, int64_t, int64_t,
                                       rmm::cuda_stream_view);
template int64_t first_zero_row<double>(const double*, int64_t, int64_t,
                                        rmm::cuda_stream_view);

}  // namespace cumlrkt
