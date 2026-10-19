#include <cuda_runtime_api.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "raftrkt/error.hpp"
#include "spectral_device.hpp"

namespace cumlrkt::cluster {

namespace {

struct scan {
  int64_t first;
  int64_t positive;
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
                        !(w >= 0) || isinf(w);
    return {faulty ? i : nnz, w > 0 ? 1 : 0};
  }
};

struct earliest {
  __device__ scan operator()(const scan& a, const scan& b) const {
    return {a.first < b.first ? a.first : b.first, a.positive | b.positive};
  }
};

template <typename T>
T read_one(const T* data, int64_t i, rmm::cuda_stream_view stream) {
  T value{};
  raftrkt::cuda_check(cudaMemcpyAsync(&value, data + i, sizeof(T),
                                      cudaMemcpyDeviceToHost, stream.value()),
                      "cudaMemcpyAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
  return value;
}

graph_fault classified(int64_t entry, int32_t r, int32_t c, float w, int64_t n,
                       bool positive) {
  if (r < 0 || r >= n) {
    return {entry, graph_fault_kind::row_index, static_cast<double>(r),
            positive};
  }
  if (c < 0 || c >= n) {
    return {entry, graph_fault_kind::col_index, static_cast<double>(c),
            positive};
  }
  if (r == c) {
    return {entry, graph_fault_kind::self_loop, static_cast<double>(r),
            positive};
  }
  return {entry, graph_fault_kind::weight, static_cast<double>(w), positive};
}

}  // namespace

graph_fault first_graph_fault(const int32_t* rows, const int32_t* cols,
                              const float* vals, int64_t nnz, int64_t n,
                              rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> entries(0);
  const scan found = thrust::transform_reduce(
      rmm::exec_policy(stream), entries, entries + nnz,
      entry_scan{rows, cols, vals, nnz, n}, scan{nnz, 0}, earliest{});
  if (found.first == nnz) {
    return {-1, graph_fault_kind::none, 0, found.positive != 0};
  }
  const int64_t i = found.first;
  return classified(i, read_one(rows, i, stream), read_one(cols, i, stream),
                    read_one(vals, i, stream), n, found.positive != 0);
}

}  // namespace cumlrkt::cluster
