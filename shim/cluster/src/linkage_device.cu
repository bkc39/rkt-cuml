#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "linkage_device.hpp"

namespace cumlrkt::cluster {

namespace {

struct squared_norm {
  const float* x;
  int64_t d;

  __device__ double operator()(int64_t i) const {
    const float* row = x + i * d;
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

struct ascending_pair {
  int32_t* children;

  __device__ void operator()(int64_t i) const {
    int32_t* pair = children + 2 * i;
    if (pair[0] > pair[1]) {
      const int32_t first = pair[0];
      pair[0] = pair[1];
      pair[1] = first;
    }
  }
};

}  // namespace

std::pair<int64_t, double> first_large_row(const float* x, int64_t n, int64_t d,
                                           double most,
                                           rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const squared_norm norm{x, d};
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

void ascending_pairs(int32_t* children, int64_t rows,
                     rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> first(0);
  thrust::for_each(rmm::exec_policy(stream), first, first + rows,
                   ascending_pair{children});
}

}  // namespace cumlrkt::cluster
