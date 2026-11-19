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
  const double* u;
  int64_t n;
  int64_t d;
  double* z;

  __device__ void operator()(int64_t i) const {
    double s = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      s = fma(static_cast<double>(x[(k * n) + i]) - mean[k], u[k], s);
    }
    z[i] = s;
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

constexpr int gram_error_bits = 14;
constexpr int64_t most_ritz = 16;

struct eigen_pairs {
  std::vector<double> values;
  std::vector<double> vectors;
};

template <typename S>
std::optional<eigen_pairs> eigen_of(const raft::handle_t& handle,
                                    const std::vector<S>& m, int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<S> a = to_device(m, stream);
  buffer<S> values = scratch<S>(d, stream);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(solver, stream.value()));
  const auto dim = static_cast<int>(d);
  constexpr auto mode = CUSOLVER_EIG_MODE_VECTOR;
  constexpr auto fill = CUBLAS_FILL_MODE_LOWER;
  int lwork = 0;
  if constexpr (std::is_same_v<S, float>) {
    RAFT_CUSOLVER_TRY(cusolverDnSsyevd_bufferSize(
        solver, mode, fill, dim, a.data(), dim, values.data(), &lwork));
  } else {
    RAFT_CUSOLVER_TRY(cusolverDnDsyevd_bufferSize(
        solver, mode, fill, dim, a.data(), dim, values.data(), &lwork));
  }
  buffer<S> work = scratch<S>(lwork, stream);
  buffer<int> info = scratch<int>(1, stream);
  if constexpr (std::is_same_v<S, float>) {
    RAFT_CUSOLVER_TRY(cusolverDnSsyevd(solver, mode, fill, dim, a.data(), dim,
                                       values.data(), work.data(), lwork,
                                       info.data()));
  } else {
    RAFT_CUSOLVER_TRY(cusolverDnDsyevd(solver, mode, fill, dim, a.data(), dim,
                                       values.data(), work.data(), lwork,
                                       info.data()));
  }
  if (to_host(info, stream)[0] != 0) {
    return std::nullopt;
  }
  const std::vector<S> l = to_host(values, stream);
  const std::vector<S> v = to_host(a, stream);
  return eigen_pairs{{l.begin(), l.end()}, {v.begin(), v.end()}};
}

double dot(const double* a, const double* b, int64_t d) {
  return std::transform_reduce(a, a + d, b, 0.0);
}

std::vector<double> orthonormal(const std::vector<double>& vectors, int64_t d,
                                int64_t k) {
  std::vector<double> v(vectors.begin(), vectors.begin() + (k * d));
  for (int64_t j = 0; j < k; ++j) {
    double* vj = v.data() + (j * d);
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t i = 0; i < j; ++i) {
        const double* vi = v.data() + (i * d);
        const double c = dot(vi, vj, d);
        std::transform(vj, vj + d, vi, vj,
                       [c](double a, double b) { return a - (c * b); });
      }
    }
    const double norm = std::sqrt(dot(vj, vj, d));
    std::transform(vj, vj + d, vj, [norm](double a) { return a / norm; });
  }
  return v;
}

std::vector<double> combined(const std::vector<double>& columns,
                             const double* c, int64_t d, int64_t k) {
  std::vector<double> y(d, 0.0);
  for (int64_t j = 0; j < k; ++j) {
    for (int64_t i = 0; i < d; ++i) {
      y[i] += c[j] * columns[(j * d) + i];
    }
  }
  return y;
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

  [[nodiscard]] std::vector<double> scaled(const double* v) const {
    std::vector<double> u(d);
    for (int64_t k = 0; k < d; ++k) {
      u[k] = v[k] * scale[k];
    }
    return u;
  }

  [[nodiscard]] buffer<double> project(const std::vector<double>& u) const {
    const rmm::cuda_stream_view stream = stream_of(handle);
    const buffer<double> on_device = to_device(u, stream);
    buffer<double> z = scratch<double>(n, stream);
    each(handle, n,
         projected_rows<T>{x, mean.data(), on_device.data(), n, d, z.data()});
    return z;
  }

  [[nodiscard]] double quotient(const double* v) const {
    const std::vector<double> u = scaled(v);
    const buffer<double> z = project(u);
    const thrust::counting_iterator<int64_t> rows(0);
    return thrust::transform_reduce(rmm::exec_policy(stream_of(handle)), rows,
                                    rows + n,
                                    weighted_square_of<T>{z.data(), w}, 0.0,
                                    thrust::plus<double>{}) +
           (alpha * dot(u.data(), u.data(), d));
  }

  [[nodiscard]] std::vector<double> applied(const double* v) const {
    const std::vector<double> u = scaled(v);
    const buffer<double> z = project(u);
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

struct ritz_pair {
  double value;
  std::vector<double> vector;
  std::vector<double> applied;
};

template <typename T>
std::optional<ritz_pair> smallest_ritz(const quotients<T>& q,
                                       const eigen_pairs& e, int64_t window) {
  const int64_t d = q.d;
  const std::vector<double> v = orthonormal(e.vectors, d, window);
  std::vector<double> av(window * d);
  for (int64_t j = 0; j < window; ++j) {
    const std::vector<double> r = q.applied(v.data() + (j * d));
    std::ranges::copy(r, av.begin() + (j * d));
  }
  std::vector<double> h(window * window);
  for (int64_t i = 0; i < window; ++i) {
    for (int64_t j = 0; j < window; ++j) {
      h[(j * window) + i] =
          0.5 * (dot(v.data() + (i * d), av.data() + (j * d), d) +
                 dot(v.data() + (j * d), av.data() + (i * d), d));
    }
  }
  const std::optional<eigen_pairs> small = eigen_of(q.handle, h, window);
  if (!small) {
    return std::nullopt;
  }
  const double* c = small->vectors.data();
  return ritz_pair{small->values.front(), combined(v, c, d, window),
                   combined(av, c, d, window)};
}

std::vector<double> refined(const eigen_pairs& e, const ritz_pair& p,
                            int64_t window, int64_t d) {
  const double gap = std::ldexp(e.values.back(), -20);
  std::vector<double> v = p.vector;
  for (int64_t j = window; j < d; ++j) {
    const double apart = e.values[j] - p.value;
    if (!(apart > gap)) {
      continue;
    }
    const double* vj = e.vectors.data() + (j * d);
    double c = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      c += vj[k] * (p.applied[k] - (p.value * p.vector[k]));
    }
    for (int64_t k = 0; k < d; ++k) {
      v[k] -= vj[k] * c / apart;
    }
  }
  const double norm = std::sqrt(dot(v.data(), v.data(), d));
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
  const std::vector<double> gram =
      lower_gram(handle, x, w, mean.data(), scales.data(), n, d);
  std::vector<float> m(gram.begin(), gram.end());
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += static_cast<float>(a * scale[k] * scale[k]);
  }
  const std::optional<eigen_pairs> e = eigen_of(handle, m, d);
  if (!e) {
    return infinite;
  }
  const quotients<T> q{handle, x, w, mean, scale, n, d, a};
  const double largest = q.quotient(
      orthonormal(std::vector<double>(e->vectors.end() - d, e->vectors.end()),
                  d, 1)
          .data());
  const double reach =
      e->values.front() + std::ldexp(e->values.back(), -gram_error_bits);
  const int64_t window = std::clamp<int64_t>(
      std::ranges::count_if(e->values,
                            [reach](double l) { return l <= reach; }),
      1, std::min(most_ritz, d));
  const std::optional<ritz_pair> p = smallest_ritz(q, *e, window);
  if (!p) {
    return infinite;
  }
  const double smallest =
      std::min(p->value, q.quotient(refined(*e, *p, window, d).data()));
  return smallest > 0 ? largest / smallest : infinite;
}

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window) {
  if (solver.family == glm_family::ols) {
    return std::is_same_v<T, float> &&
           (solver.algorithm == CUML_LINEAR_MODEL_OLS_QR ||
            solver.algorithm == CUML_LINEAR_MODEL_OLS_SVD_QR);
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
      "X: the {} solve's {} has u*kappa {} >= 2^-{}; {}", solver_name(solver),
      ols ? "X" : "regularised Gram matrix", printed(share), tolerance_bits<T>,
      ols ? "X may have collinear columns"
      : std::is_same_v<T, float>
          ? "X is ill-conditioned for this alpha (float64 or a larger "
            "alpha may fit)"
          : "X is ill-conditioned for this alpha (a larger alpha may "
            "fit)"));
}

template bool checks_conditioning<float>(glm_solver, bool);
template bool checks_conditioning<double>(glm_solver, bool);
template void require_conditioned<float>(double, glm_solver);
template void require_conditioned<double>(double, glm_solver);
template double equilibrated_condition<float>(const raft::handle_t&,
                                              const float*, const float*,
                                              int64_t, int64_t, float, bool);
template double equilibrated_condition<double>(const raft::handle_t&,
                                               const double*, const double*,
                                               int64_t, int64_t, double, bool);

}  // namespace cumlrkt::glm
