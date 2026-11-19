#include <cublas_v2.h>
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
#include <optional>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
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
#include "ritz.hpp"
#include "symmetric_eigen.hpp"

namespace cumlrkt::glm {

namespace {

using ritz::combined;
using ritz::dot;
using ritz::eigen_pairs;
using ritz::exact_gram_cheaper;
using ritz::reach_of;
using ritz::refined;
using ritz::ritz_pair;
using ritz::symmetric_block;
using ritz::window_of;

constexpr int64_t block_rows = ritz::gram_block_rows;
constexpr int64_t most_blocks = 64;
constexpr int64_t block_floats = int64_t{1} << 22;
constexpr int64_t column_parts = 1024;
constexpr double infinite = std::numeric_limits<double>::infinity();

struct part_of {
  int64_t parts;

  __device__ int64_t operator()(int64_t o) const {
    return o / parts;
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

template <typename T, typename U>
struct equilibrated_rows {
  const T* x;
  const T* w;
  const double* mean;
  const double* scale;
  int64_t n;
  int64_t start;
  int64_t rows;
  U* out;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / rows;
    const int64_t i = start + o - (k * rows);
    if (i >= n) {
      out[o] = U{0};
      return;
    }
    const double root = w == nullptr ? 1.0 : sqrt(static_cast<double>(w[i]));
    out[o] = static_cast<U>(root * scale[k] *
                            (static_cast<double>(x[(k * n) + i]) - mean[k]));
  }
};

template <typename T>
struct projected_rows {
  const T* x;
  const T* w;
  const double* mean;
  const double* u;
  int64_t n;
  int64_t d;
  double* z;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / n;
    const int64_t i = o - (j * n);
    double s = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      s = fma(static_cast<double>(x[(k * n) + i]) - mean[k], u[(j * d) + k], s);
    }
    z[o] = (w == nullptr ? 1.0 : static_cast<double>(w[i])) * s;
  }
};

template <typename T>
struct weighted_product {
  const T* x;
  const double* mean;
  const double* z;
  int64_t n;
  int64_t d;

  __device__ double operator()(int64_t o, int64_t i) const {
    const int64_t j = o / d;
    const int64_t k = o - (j * d);
    return (static_cast<double>(x[(k * n) + i]) - mean[k]) * z[(j * n) + i];
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
  const int64_t count = std::clamp(block_floats / d, int64_t{1}, column_parts);
  buffer<double> parts = scratch<double>(d * count, stream);
  each(handle, d * count, partial_sums<F>{entry, n, count, parts.data()});
  const thrust::counting_iterator<int64_t> cells(0);
  buffer<double> sums = scratch<double>(d, stream);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(cells, part_of{count}),
      thrust::make_transform_iterator(cells + (d * count), part_of{count}),
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
struct equilibration {
  const raft::handle_t& handle;
  const T* x;
  const T* w;
  const buffer<double>& mean;
  const buffer<double>& scales;
  const std::vector<double>& scale;
  int64_t n;
  int64_t d;
  double alpha;

  template <typename U, typename F>
  void each_block(int64_t rows, F step) const {
    buffer<U> block = scratch<U>(rows * d, stream_of(handle));
    for (int64_t start = 0; start < n; start += rows) {
      each(handle, rows * d,
           equilibrated_rows<T, U>{x, w, mean.data(), scales.data(), n, start,
                                   rows, block.data()});
      step(block.data(), start);
    }
  }
};

cublasHandle_t cublas_on(const raft::handle_t& handle) {
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream_of(handle).value()));
  return cublas;
}

template <typename T>
std::vector<double> lower_gram(const equilibration<T>& q) {
  const rmm::cuda_stream_view stream = stream_of(q.handle);
  const int64_t cells = q.d * q.d;
  buffer<double> gram = to_device(std::vector<double>(cells, 0.0), stream);
  const int64_t blocks =
      std::min({most_blocks, std::max(int64_t{1}, block_floats / cells),
                (q.n + block_rows - 1) / block_rows});
  const int64_t rows = blocks * block_rows;
  buffer<float> parts = scratch<float>(blocks * cells, stream);
  cublasHandle_t cublas = cublas_on(q.handle);
  const float one = 1.0F;
  const float zero = 0.0F;
  const auto dim = static_cast<int>(q.d);
  q.template each_block<float>(rows, [&](const float* block, int64_t start) {
    const int64_t count =
        std::min(blocks, (q.n - start + block_rows - 1) / block_rows);
    RAFT_CUBLAS_TRY(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim,
        static_cast<int>(block_rows), &one, block, static_cast<int>(rows),
        block_rows, block, static_cast<int>(rows), block_rows, &zero,
        parts.data(), dim, cells, static_cast<int>(count)));
    each(q.handle, cells, added_into{parts.data(), count, cells, gram.data()});
  });
  return to_host(gram, stream);
}

template <typename T>
std::vector<double> applied(const equilibration<T>& q,
                            const std::vector<double>& v, int64_t c) {
  const rmm::cuda_stream_view stream = stream_of(q.handle);
  const int64_t d = q.d;
  std::vector<double> u(v.size());
  for (int64_t o = 0; o < d * c; ++o) {
    u[o] = v[o] * q.scale[o % d];
  }
  const buffer<double> on_device = to_device(u, stream);
  buffer<double> z = scratch<double>(q.n * c, stream);
  each(q.handle, q.n * c,
       projected_rows<T>{q.x, q.w, q.mean.data(), on_device.data(), q.n, d,
                         z.data()});
  std::vector<double> out = to_host(
      column_sums(q.handle,
                  weighted_product<T>{q.x, q.mean.data(), z.data(), q.n, d},
                  q.n, d * c),
      stream);
  for (int64_t o = 0; o < d * c; ++o) {
    out[o] = q.scale[o % d] * (out[o] + (q.alpha * u[o]));
  }
  return out;
}

std::vector<double> crossed(const raft::handle_t& handle,
                            const std::vector<double>& a,
                            const std::vector<double>& b, int64_t d,
                            int64_t c) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const buffer<double> da = to_device(a, stream);
  const buffer<double> db = to_device(b, stream);
  buffer<double> out = scratch<double>(c * c, stream);
  const double one = 1.0;
  const double zero = 0.0;
  const auto [k, m] = std::pair{static_cast<int>(d), static_cast<int>(c)};
  RAFT_CUBLAS_TRY(cublasDgemm(cublas_on(handle), CUBLAS_OP_T, CUBLAS_OP_N, m, m,
                              k, &one, da.data(), k, db.data(), k, &zero,
                              out.data(), m));
  return to_host(out, stream);
}

template <typename T>
double exact_condition(const equilibration<T>& q) {
  const rmm::cuda_stream_view stream = stream_of(q.handle);
  const int64_t d = q.d;
  const int64_t rows = std::min(q.n, std::max(int64_t{1}, block_floats / d));
  buffer<double> g = to_device(std::vector<double>(d * d, 0.0), stream);
  cublasHandle_t cublas = cublas_on(q.handle);
  const double one = 1.0;
  q.template each_block<double>(rows, [&](const double* b, int64_t) {
    RAFT_CUBLAS_TRY(cublasDsyrk(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                static_cast<int>(d), static_cast<int>(rows),
                                &one, b, static_cast<int>(rows), &one, g.data(),
                                static_cast<int>(d)));
  });
  std::vector<double> m = to_host(g, stream);
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += q.alpha * q.scale[k] * q.scale[k];
  }
  const std::optional<eigen_pairs> e = eigen_of(q.handle, m, d, false);
  if (!e || !(e->values.front() > 0)) {
    return infinite;
  }
  return e->values.back() / e->values.front();
}

template <typename T>
double ritz_condition(const equilibration<T>& q, const eigen_pairs& e,
                      int64_t window) {
  const int64_t d = q.d;
  const int64_t c = window + 1;
  std::vector<double> basis(e.vectors.begin(),
                            e.vectors.begin() + (window * d));
  basis.insert(basis.end(), e.vectors.end() - d, e.vectors.end());
  const std::vector<double> av = applied(q, basis, c);
  const std::vector<double> h = crossed(q.handle, basis, av, d, c);
  const std::vector<double> s = crossed(q.handle, basis, basis, d, c);
  const std::optional<eigen_pairs> small =
      pencil_of(q.handle, symmetric_block(h, c, window),
                symmetric_block(s, c, window), window);
  if (!small) {
    return infinite;
  }
  const double* y = small->vectors.data();
  const ritz_pair p{small->values.front(), combined(basis, d, y, window),
                    combined(av, d, y, window)};
  const double largest = h.back() / s.back();
  if (e.values.front() > reach_of(d)) {
    return p.value > 0 ? largest / p.value : infinite;
  }
  const std::vector<double> v = refined(e, window, p, d);
  const std::vector<double> rv = applied(q, v, 1);
  const double smallest = std::min(p.value, dot(v.data(), rv.data(), d));
  return smallest > 0 ? largest / smallest : infinite;
}

}  // namespace

template <typename T>
double equilibrated_condition(const raft::handle_t& handle, const T* x,
                              const T* w, int64_t n, int64_t d, T alpha,
                              bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto a = static_cast<double>(alpha);
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
  const equilibration<T> q{handle, x, w, mean, scales, scale, n, d, a};
  if (exact_gram_cheaper(1, d)) {
    return exact_condition(q);
  }
  const std::vector<double> gram = lower_gram(q);
  std::vector<float> m(gram.begin(), gram.end());
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += static_cast<float>(a * scale[k] * scale[k]);
  }
  const std::optional<eigen_pairs> e = eigen_of(handle, m, d, true);
  if (!e) {
    return infinite;
  }
  const int64_t window = window_of(e->values, d);
  return exact_gram_cheaper(window, d) ? exact_condition(q)
                                       : ritz_condition(q, *e, window);
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
