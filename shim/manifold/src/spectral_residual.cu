#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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
    const double x = u[o];
    const double y = lu[o];
    if (!normalised) {
      return {x * x, x * y, y * y, x, x};
    }
    const double d = degree[o % n];
    if (!(d > 0)) {
      return {x * x, x * x, x * x, DBL_MAX, -DBL_MAX};
    }
    return {d * x * x, x * y, y * y / d, x, x};
  }
};

struct added {
  __device__ sums operator()(const sums& a, const sums& b) const {
    return {a.uu + b.uu, a.ulu + b.ulu, a.lul + b.lul, fmin(a.lo, b.lo),
            fmax(a.hi, b.hi)};
  }
};

struct weight_ends {
  __device__ thrust::pair<double, double> operator()(double w) const {
    return {w > 0 ? w : DBL_MAX, w};
  }
};

struct widest {
  __device__ thrust::pair<double, double> operator()(
      const thrust::pair<double, double>& a,
      const thrust::pair<double, double>& b) const {
    return {fmin(a.first, b.first), fmax(a.second, b.second)};
  }
};

struct directed_pair {
  coo c;
  int64_t n;

  __device__ int64_t operator()(int64_t e) const {
    const int64_t r = c.rows[e];
    const int64_t s = c.cols[e];
    const int64_t lo = r < s ? r : s;
    const int64_t hi = r < s ? s : r;
    return (((lo * n) + hi) * 2) + (r > s ? 1 : 0);
  }
};

struct widened {
  __device__ double operator()(float w) const {
    return w;
  }
};

struct signed_by_direction {
  __device__ double operator()(int64_t key, double w) const {
    const double single = static_cast<float>(w);
    return key % 2 != 0 ? -single : single;
  }
};

struct unordered_pair {
  __device__ int64_t operator()(int64_t key) const {
    return key / 2;
  }
};

struct nonzero {
  __device__ bool operator()(double w) const {
    return w != 0;
  }
};

constexpr double residual_bound = 1e-3;
constexpr double constant_spread = 1e-3;

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
    const double top = std::max(std::abs(s.hi), std::abs(s.lo));
    fits.push_back({std::sqrt(off / s.uu) / (normalised ? 1.0 : most),
                    s.hi >= s.lo && top > 0 ? (s.hi - s.lo) / top : 1.0});
  }
  return fits;
}

std::optional<refusal> first_refused(const std::vector<column_fit>& fits,
                                     bool drop_first) {
  for (std::size_t j = 0; j < fits.size(); ++j) {
    const bool constant = drop_first && fits[j].spread < constant_spread;
    if (!(fits[j].residual <= residual_bound) || constant) {
      return refusal{.column = j, .constant = constant};
    }
  }
  return std::nullopt;
}

graph_shape shape_of(const raft::handle_t& handle, const coo& c, int64_t n) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  const auto policy = rmm::exec_policy(stream);
  const auto count = static_cast<std::size_t>(c.nnz);
  rmm::device_uvector<int64_t> keys(count, stream);
  rmm::device_uvector<double> weights(count, stream);
  rmm::device_uvector<int64_t> pairs(count, stream);
  rmm::device_uvector<double> summed(count, stream);
  const thrust::counting_iterator<int64_t> entries(0);
  thrust::transform(policy, entries, entries + c.nnz, keys.begin(),
                    directed_pair{c, n});
  thrust::transform(policy, c.vals, c.vals + c.nnz, weights.begin(), widened{});
  thrust::sort_by_key(policy, keys.begin(), keys.end(), weights.begin());
  const auto distinct =
      thrust::reduce_by_key(policy, keys.begin(), keys.end(), weights.begin(),
                            pairs.begin(), summed.begin())
          .first;
  const auto last = summed.begin() + (distinct - pairs.begin());
  const thrust::pair<double, double> ends = thrust::transform_reduce(
      policy, summed.begin(), last, weight_ends{},
      thrust::pair<double, double>{DBL_MAX, 0.0}, widest{});
  thrust::transform(policy, pairs.begin(), distinct, summed.begin(),
                    summed.begin(), signed_by_direction{});
  const auto nets = thrust::reduce_by_key(
      policy, thrust::make_transform_iterator(pairs.begin(), unordered_pair{}),
      thrust::make_transform_iterator(distinct, unordered_pair{}),
      summed.begin(), thrust::make_discard_iterator(), weights.begin());
  const bool symmetric =
      thrust::none_of(policy, weights.begin(), nets.second, nonzero{});
  return {.symmetric = symmetric, .span = ends.second / ends.first};
}

}  // namespace cumlrkt::manifold
