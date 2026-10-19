#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "spectral_device.hpp"

namespace cumlrkt::cluster {

namespace {

struct scan {
  int64_t first;
  int64_t positive;
  double total;
};

struct entry_scan {
  const int32_t* rows;
  const int32_t* cols;
  const float* vals;
  int64_t nnz;
  int64_t n;

  __device__ scan operator()(int64_t i) const {
    const int64_t r = rows[i];
    const int64_t c = cols[i];
    const float w = vals[i];
    const bool faulty = r < 0 || r >= n || c < 0 || c >= n || r == c ||
                        !(w >= 0) || isinf(w) || (w > 0 && w < FLT_MIN);
    return {faulty ? i : nnz, w > 0 ? 1 : 0, static_cast<double>(w)};
  }
};

struct earliest {
  __device__ scan operator()(const scan& a, const scan& b) const {
    return {a.first < b.first ? a.first : b.first, a.positive | b.positive,
            a.total + b.total};
  }
};

}  // namespace

graph_scan scan_graph(const int32_t* rows, const int32_t* cols,
                      const float* vals, int64_t nnz, int64_t n,
                      rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> entries(0);
  const scan found = thrust::transform_reduce(
      rmm::exec_policy(stream), entries, entries + nnz,
      entry_scan{rows, cols, vals, nnz, n}, scan{nnz, 0, 0}, earliest{});
  return {found.first == nnz ? -1 : found.first, found.positive != 0,
          found.total};
}

}  // namespace cumlrkt::cluster
