#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "gram_condition.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int64_t block_rows = 1024;
constexpr int64_t most_blocks = 64;
constexpr int64_t block_floats = int64_t{1} << 22;
constexpr int64_t column_parts = 1024;

struct part_of {
  __device__ int64_t operator()(int64_t o) const {
    return o / column_parts;
  }
};

template <typename T>
struct weight_of {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return w == nullptr ? 1.0 : static_cast<double>(w[i]);
  }
};

template <typename T>
struct weighted_entry {
  const T* x;
  const T* w;
  int64_t n;

  __device__ double operator()(int64_t k, int64_t i) const {
    const double v = static_cast<double>(x[(k * n) + i]);
    return w == nullptr ? v : static_cast<double>(w[i]) * v;
  }
};

template <typename T>
struct weighted_square {
  const T* x;
  const T* w;
  const double* mean;
  int64_t n;

  __device__ double operator()(int64_t k, int64_t i) const {
    const double v = static_cast<double>(x[(k * n) + i]) - mean[k];
    return (w == nullptr ? 1.0 : static_cast<double>(w[i])) * v * v;
  }
};

template <typename T>
struct equilibrated_rows {
  const T* x;
  const T* w;
  const double* mean;
  const double* scale;
  int64_t n;
  int64_t d;
  int64_t start;
  int64_t rows;
  float* out;

  __device__ void operator()(int64_t r) const {
    const int64_t i = start + r;
    if (i >= n) {
      for (int64_t k = 0; k < d; ++k) {
        out[(k * rows) + r] = 0.0F;
      }
      return;
    }
    const double root = w == nullptr ? 1.0 : sqrt(static_cast<double>(w[i]));
    for (int64_t k = 0; k < d; ++k) {
      out[(k * rows) + r] = static_cast<float>(
          root * scale[k] * (static_cast<double>(x[(k * n) + i]) - mean[k]));
    }
  }
};

struct divided_by {
  double* v;
  double total;

  __device__ void operator()(int64_t k) const {
    v[k] /= total;
  }
};

struct added_into {
  const float* parts;
  int64_t count;
  int64_t cells;
  double* gram;

  __device__ void operator()(int64_t o) const {
    double s = gram[o];
    for (int64_t b = 0; b < count; ++b) {
      s += static_cast<double>(parts[(b * cells) + o]);
    }
    gram[o] = s;
  }
};

template <typename T>
struct projected_rows {
  const T* x;
  const double* mean;
  const double* lo;
  const double* hi;
  int64_t n;
  int64_t d;
  double* z_lo;
  double* z_hi;

  __device__ void operator()(int64_t i) const {
    double s_lo = 0.0;
    double s_hi = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      const double v = static_cast<double>(x[(k * n) + i]) - mean[k];
      s_lo = fma(v, lo[k], s_lo);
      s_hi = fma(v, hi[k], s_hi);
    }
    z_lo[i] = s_lo;
    z_hi[i] = s_hi;
  }
};

template <typename T>
struct weighted_square_of {
  const double* z;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return (w == nullptr ? 1.0 : static_cast<double>(w[i])) * z[i] * z[i];
  }
};

template <typename T>
struct weighted_product {
  const T* x;
  const T* w;
  const double* mean;
  const double* z;
  int64_t n;

  __device__ double operator()(int64_t k, int64_t i) const {
    const double v = static_cast<double>(x[(k * n) + i]) - mean[k];
    return (w == nullptr ? 1.0 : static_cast<double>(w[i])) * v * z[i];
  }
};

template <typename F>
struct partial_sums {
  F entry;
  int64_t n;
  double* out;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / column_parts;
    double s = 0.0;
    for (int64_t i = o - (k * column_parts); i < n; i += column_parts) {
      s += entry(k, i);
    }
    out[o] = s;
  }
};

template <typename F>
buffer<double> column_sums(const raft::handle_t& handle, F entry, int64_t n,
                           int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> parts = scratch<double>(d * column_parts, stream);
  each(handle, d * column_parts, partial_sums<F>{entry, n, parts.data()});
  const thrust::counting_iterator<int64_t> cells(0);
  buffer<double> sums = scratch<double>(d, stream);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(cells, part_of{}),
      thrust::make_transform_iterator(cells + (d * column_parts), part_of{}),
      parts.begin(), thrust::make_discard_iterator(), sums.begin());
  return sums;
}

template <typename T>
buffer<double> column_means(const raft::handle_t& handle, const T* x,
                            const T* w, int64_t n, int64_t d,
                            bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (!fit_intercept) {
    return to_device(std::vector<double>(d, 0.0), stream);
  }
  const thrust::counting_iterator<int64_t> rows(0);
  const double total =
      thrust::transform_reduce(rmm::exec_policy(stream), rows, rows + n,
                               weight_of<T>{w}, 0.0, thrust::plus<double>{});
  buffer<double> sums = column_sums(handle, weighted_entry<T>{x, w, n}, n, d);
  each(handle, d, divided_by{sums.data(), total});
  return sums;
}

template <typename T>
std::vector<double> lower_gram(const raft::handle_t& handle, const T* x,
                               const T* w, const double* mean,
                               const double* scale, int64_t n, int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t cells = d * d;
  buffer<double> gram = to_device(std::vector<double>(cells, 0.0), stream);
  const int64_t blocks =
      std::min({most_blocks, std::max(int64_t{1}, block_floats / cells),
                (n + block_rows - 1) / block_rows});
  const int64_t rows = blocks * block_rows;
  buffer<float> block = scratch<float>(rows * d, stream);
  buffer<float> parts = scratch<float>(blocks * cells, stream);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream.value()));
  const float one = 1.0F;
  const float zero = 0.0F;
  const auto dim = static_cast<int>(d);
  for (int64_t start = 0; start < n; start += rows) {
    const int64_t count =
        std::min(blocks, (n - start + block_rows - 1) / block_rows);
    each(handle, count * block_rows,
         equilibrated_rows<T>{x, w, mean, scale, n, d, start, rows,
                              block.data()});
    RAFT_CUBLAS_TRY(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim,
        static_cast<int>(block_rows), &one, block.data(),
        static_cast<int>(rows), block_rows, block.data(),
        static_cast<int>(rows), block_rows, &zero, parts.data(), dim, cells,
        static_cast<int>(count)));
    each(handle, cells, added_into{parts.data(), count, cells, gram.data()});
  }
  return to_host(gram, stream);
}

struct eigen_pairs {
  std::vector<double> values;
  std::vector<double> vectors;
};

std::optional<eigen_pairs> eigen_of(const raft::handle_t& handle,
                                    const std::vector<double>& m, int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> a = to_device(m, stream);
  buffer<double> values = scratch<double>(d, stream);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(solver, stream.value()));
  int lwork = 0;
  RAFT_CUSOLVER_TRY(cusolverDnDsyevd_bufferSize(
      solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(d), a.data(), static_cast<int>(d), values.data(),
      &lwork));
  buffer<double> work = scratch<double>(lwork, stream);
  buffer<int> info = scratch<int>(1, stream);
  RAFT_CUSOLVER_TRY(
      cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                       static_cast<int>(d), a.data(), static_cast<int>(d),
                       values.data(), work.data(), lwork, info.data()));
  if (to_host(info, stream)[0] != 0) {
    return std::nullopt;
  }
  return eigen_pairs{to_host(values, stream), to_host(a, stream)};
}

template <typename T>
struct quotients {
  const raft::handle_t& handle;
  const T* x;
  const T* w;
  const buffer<double>& mean;
  const std::vector<double>& scale;
  int64_t n;
  int64_t d;
  double alpha;

  [[nodiscard]] double penalty(const std::vector<double>& u) const {
    double p = 0.0;
    for (const double v : u) {
      p += alpha * v * v;
    }
    return p;
  }

  [[nodiscard]] std::vector<double> scaled(const double* v) const {
    std::vector<double> u(d);
    for (int64_t k = 0; k < d; ++k) {
      u[k] = v[k] * scale[k];
    }
    return u;
  }

  [[nodiscard]] double sum_of(const buffer<double>& z) const {
    const thrust::counting_iterator<int64_t> rows(0);
    return thrust::transform_reduce(
        rmm::exec_policy(stream_of(handle)), rows, rows + n,
        weighted_square_of<T>{z.data(), w}, 0.0, thrust::plus<double>{});
  }

  std::pair<buffer<double>, buffer<double>> project(
      const std::vector<double>& lo, const std::vector<double>& hi) const {
    const rmm::cuda_stream_view stream = stream_of(handle);
    const buffer<double> l = to_device(lo, stream);
    const buffer<double> h = to_device(hi, stream);
    buffer<double> z_lo = scratch<double>(n, stream);
    buffer<double> z_hi = scratch<double>(n, stream);
    each(handle, n,
         projected_rows<T>{x, mean.data(), l.data(), h.data(), n, d,
                           z_lo.data(), z_hi.data()});
    return {std::move(z_lo), std::move(z_hi)};
  }

  [[nodiscard]] std::vector<double> applied(
      const buffer<double>& z, const std::vector<double>& u) const {
    std::vector<double> r = to_host(
        column_sums(handle, weighted_product<T>{x, w, mean.data(), z.data(), n},
                    n, d),
        stream_of(handle));
    for (int64_t k = 0; k < d; ++k) {
      r[k] = scale[k] * (r[k] + (alpha * u[k]));
    }
    return r;
  }
};

std::vector<double> refined(const eigen_pairs& e, const std::vector<double>& r,
                            double mu, int64_t d) {
  const double gap = std::ldexp(e.values.back(), -20);
  std::vector<double> v(e.vectors.begin(), e.vectors.begin() + d);
  for (int64_t j = 1; j < d; ++j) {
    const double apart = e.values[j] - mu;
    if (!(apart > gap)) {
      continue;
    }
    const double* vj = e.vectors.data() + (j * d);
    double c = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      c += vj[k] * (r[k] - (mu * e.vectors[k]));
    }
    for (int64_t k = 0; k < d; ++k) {
      v[k] -= vj[k] * c / apart;
    }
  }
  const double norm =
      std::sqrt(std::transform_reduce(v.begin(), v.end(), v.begin(), 0.0));
  std::ranges::transform(v, v.begin(), [norm](double x) { return x / norm; });
  return v;
}

}  // namespace

template <typename T>
double equilibrated_condition(const raft::handle_t& handle, const T* x,
                              const T* w, int64_t n, int64_t d, T alpha,
                              bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto a = static_cast<double>(alpha);
  constexpr double infinite = std::numeric_limits<double>::infinity();
  const buffer<double> mean = column_means(handle, x, w, n, d, fit_intercept);
  const std::vector<double> spread = to_host(
      column_sums(handle, weighted_square<T>{x, w, mean.data(), n}, n, d),
      stream);
  if (std::ranges::any_of(spread, [a](double s) { return !(s + a > 0); })) {
    return infinite;
  }
  std::vector<double> scale(d);
  std::ranges::transform(spread, scale.begin(),
                         [a](double s) { return 1.0 / std::sqrt(s + a); });
  const buffer<double> scales = to_device(scale, stream);
  std::vector<double> m =
      lower_gram(handle, x, w, mean.data(), scales.data(), n, d);
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += a * scale[k] * scale[k];
  }
  const std::optional<eigen_pairs> e = eigen_of(handle, m, d);
  if (!e) {
    return infinite;
  }
  const quotients<T> q{handle, x, w, mean, scale, n, d, a};
  const std::vector<double> u_lo = q.scaled(e->vectors.data());
  const std::vector<double> u_hi = q.scaled(e->vectors.data() + ((d - 1) * d));
  const auto [z_lo, z_hi] = q.project(u_lo, u_hi);
  const double largest = q.sum_of(z_hi) + q.penalty(u_hi);
  const std::vector<double> r = q.applied(z_lo, u_lo);
  const double mu =
      std::transform_reduce(r.begin(), r.end(), e->vectors.begin(), 0.0);
  const std::vector<double> u = q.scaled(refined(*e, r, mu, d).data());
  const buffer<double> z = std::move(q.project(u, u).first);
  const double smallest = std::min(mu, q.sum_of(z) + q.penalty(u));
  return smallest > 0 ? largest / smallest : infinite;
}

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window,
                         bool forced_svd) {
  if (solver.family == glm_family::ols) {
    return std::is_same_v<T, float> && !forced_svd;
  }
  return solver.family == glm_family::ridge && alpha_window &&
         solver.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD;
}

template <typename T>
void require_conditioned(double kappa, glm_solver solver) {
  const bool ols = solver.family == glm_family::ols;
  const double share = unit_roundoff<T> * (ols ? std::sqrt(kappa) : kappa);
  if (share < std::ldexp(1.0, -tolerance_bits<T>)) {
    return;
  }
  throw logic_error(std::format(
      "X: the {} solve's {}Gram matrix has u*kappa {} >= 2^-{}; {}",
      solver_name(solver), ols ? "" : "regularised ", printed(share),
      tolerance_bits<T>,
      ols ? "X may have collinear columns"
      : std::is_same_v<T, float>
          ? "X is ill-conditioned for this alpha (float64 or a larger "
            "alpha may fit)"
          : "X is ill-conditioned for this alpha (a larger alpha may "
            "fit)"));
}

template bool checks_conditioning<float>(glm_solver, bool, bool);
template bool checks_conditioning<double>(glm_solver, bool, bool);
template void require_conditioned<float>(double, glm_solver);
template void require_conditioned<double>(double, glm_solver);
template double equilibrated_condition<float>(const raft::handle_t&,
                                              const float*, const float*,
                                              int64_t, int64_t, float, bool);
template double equilibrated_condition<double>(const raft::handle_t&,
                                               const double*, const double*,
                                               int64_t, int64_t, double, bool);

}  // namespace cumlrkt::glm
