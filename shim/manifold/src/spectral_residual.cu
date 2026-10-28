#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/graph.hpp"
#include "raftrkt/error.hpp"
#include "spectral_residual.hpp"

namespace cumlrkt::manifold {

namespace {

struct degree_sum {
  const int32_t* rows;
  const float* vals;
  double* degree;

  __device__ void operator()(int64_t e) const {
    atomicAdd(degree + rows[e], static_cast<double>(vals[e]));
  }
};

struct laplacian_sum {
  coo c;
  const float* u;
  double* lu;
  int64_t n;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / c.nnz;
    const int64_t e = o - (j * c.nnz);
    const int64_t r = c.rows[e];
    const float* x = u + (j * n);
    const double step =
        static_cast<double>(x[r]) - static_cast<double>(x[c.cols[e]]);
    atomicAdd(lu + (j * n) + r, static_cast<double>(c.vals[e]) * step);
  }
};

struct sums {
  double uu;
  double ulu;
  double lul;
  double lo;
  double hi;
};

struct column_terms {
  const float* u;
  const double* lu;
  const double* degree;
  int64_t n;
  bool normalised;

  __device__ sums operator()(int64_t o) const {
    const double m = normalised ? degree[o % n] : 1.0;
    const double x = u[o];
    if (!(m > 0)) {
      return {0, 0, 0, x, x};
    }
    const double y = lu[o];
    return {m * x * x, x * y, y * y / m, x, x};
  }
};

struct added {
  __device__ sums operator()(const sums& a, const sums& b) const {
    return {a.uu + b.uu, a.ulu + b.ulu, a.lul + b.lul, fmin(a.lo, b.lo),
            fmax(a.hi, b.hi)};
  }
};

struct weight_ends {
  __device__ thrust::pair<float, float> operator()(float w) const {
    return {w > 0 ? w : FLT_MAX, w};
  }
};

struct widest {
  __device__ thrust::pair<float, float> operator()(
      const thrust::pair<float, float>& a,
      const thrust::pair<float, float>& b) const {
    return {fminf(a.first, b.first), fmaxf(a.second, b.second)};
  }
};

struct column_of {
  int64_t n;

  __device__ int64_t operator()(int64_t o) const {
    return o / n;
  }
};

}  // namespace

std::vector<column_fit> column_fits(const raft::handle_t& handle, const coo& c,
                                    int64_t n, const float* columns, int64_t k,
                                    bool normalised) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<double> degree(static_cast<std::size_t>(n), stream);
  rmm::device_uvector<double> lu(static_cast<std::size_t>(n * k), stream);
  rmm::device_uvector<sums> found(static_cast<std::size_t>(k), stream);
  thrust::fill_n(rmm::exec_policy(stream), degree.data(), n, 0.0);
  thrust::fill_n(rmm::exec_policy(stream), lu.data(), n * k, 0.0);
  each(handle, c.nnz, degree_sum{c.rows, c.vals, degree.data()});
  each(handle, c.nnz * k, laplacian_sum{c, columns, lu.data(), n});
  const thrust::counting_iterator<int64_t> entries(0);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(entries, column_of{n}),
      thrust::make_transform_iterator(entries + (n * k), column_of{n}),
      thrust::make_transform_iterator(
          entries,
          column_terms{columns, lu.data(), degree.data(), n, normalised}),
      thrust::make_discard_iterator(), found.data(),
      thrust::equal_to<int64_t>{}, added{});
  const double most =
      thrust::reduce(rmm::exec_policy(stream), degree.data(), degree.data() + n,
                     0.0, thrust::maximum<double>{});
  std::vector<sums> host(static_cast<std::size_t>(k));
  raftrkt::cuda_check(
      cudaMemcpyAsync(host.data(), found.data(), sizeof(sums) * host.size(),
                      cudaMemcpyDeviceToHost, stream.value()),
      "cudaMemcpyAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
  std::vector<column_fit> fits;
  fits.reserve(host.size());
  for (const sums& s : host) {
    const double rho = s.ulu / s.uu;
    const double off = std::max(0.0, s.lul - (rho * s.ulu));
    fits.push_back({std::sqrt(off / s.uu) / (normalised ? 1.0 : most),
                    (s.hi - s.lo) / std::max(std::abs(s.hi), std::abs(s.lo))});
  }
  return fits;
}

double weight_span(const coo& c, rmm::cuda_stream_view stream) {
  const thrust::pair<float, float> ends = thrust::transform_reduce(
      rmm::exec_policy(stream), c.vals, c.vals + c.nnz, weight_ends{},
      thrust::pair<float, float>{FLT_MAX, 0.0F}, widest{});
  return static_cast<double>(ends.second) / static_cast<double>(ends.first);
}

}  // namespace cumlrkt::manifold
