#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "cumlrkt/device.cuh"
#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

constexpr int64_t column_part_rows = 256;
constexpr int64_t column_part_cells = int64_t{1} << 22;

struct part_of {
  int64_t parts;

  __device__ int64_t operator()(int64_t o) const {
    return o / parts;
  }
};

template <typename F>
struct partial_sums {
  F entry;
  int64_t n;
  int64_t parts;
  double* out;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / parts;
    double s = 0.0;
    for (int64_t i = o - (k * parts); i < n; i += parts) {
      s += entry(k, i);
    }
    out[o] = s;
  }
};

template <typename F>
buffer<double> column_sums(const raft::handle_t& handle, F entry, int64_t n,
                           int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t count =
      std::clamp((n + column_part_rows - 1) / column_part_rows, int64_t{1},
                 std::max(int64_t{1}, column_part_cells / d));
  buffer<double> parts = scratch<double>(d * count, stream);
  each(handle, d * count, partial_sums<F>{entry, n, count, parts.data()});
  if (count == 1) {
    return parts;
  }
  const thrust::counting_iterator<int64_t> cells(0);
  buffer<double> sums = scratch<double>(d, stream);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(cells, part_of{count}),
      thrust::make_transform_iterator(cells + (d * count), part_of{count}),
      parts.begin(), thrust::make_discard_iterator(), sums.begin());
  return sums;
}

}  // namespace cumlrkt::glm
