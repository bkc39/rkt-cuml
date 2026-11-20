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

#include "block_gram.hpp"
#include "column_sums.cuh"
#include "cumlrkt/device.cuh"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "gram_condition.hpp"
#include "ritz.hpp"
#include "spectrum_enclosure.hpp"
#include "symmetric_eigen.hpp"

namespace cumlrkt::glm {

namespace {

using ritz::combined;
using ritz::dot;
using ritz::eigen_pairs;
using ritz::gram_path;
using ritz::gram_plan;
using ritz::plan_of;
using ritz::refined;
using ritz::ritz_pair;
using ritz::symmetric_block;

constexpr int64_t block_rows = ritz::gram_block_rows;
constexpr int64_t block_floats = ritz::block_floats;
constexpr double infinite = std::numeric_limits<double>::infinity();
constexpr double mean_refinement = 8.0;

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
struct weighted_deviation {
  const T* x;
  const T* w;
  const double* mean;
  int64_t n;

  __device__ double operator()(int64_t k, int64_t i) const {
    const double v = static_cast<double>(x[(k * n) + i]) - mean[k];
    return (w == nullptr ? 1.0 : static_cast<double>(w[i])) * v;
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

template <typename P>
struct block_part {
  const P* parts;
  int64_t cells;

  __device__ double operator()(int64_t o, int64_t b) const {
    return static_cast<double>(parts[(b * cells) + o]);
  }
};

struct added_into {
  const double* sums;
  double* gram;

  __device__ void operator()(int64_t o) const {
    gram[o] += sums[o];
  }
};

struct centred {
  buffer<double> mean;
  ritz::centring host;
};

template <typename T>
centred column_means(const raft::handle_t& handle, const T* x, const T* w,
                     int64_t n, int64_t d, bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (!fit_intercept) {
    return {to_device(std::vector<double>(d, 0.0), stream), {{}, 0.0, {}}};
  }
  const thrust::counting_iterator<int64_t> rows(0);
  const double total =
      thrust::transform_reduce(rmm::exec_policy(stream), rows, rows + n,
                               weight_of<T>{w}, 0.0, thrust::plus<double>{});
  buffer<double> sums = column_sums(handle, weighted_entry<T>{x, w, n}, n, d);
  each(handle, d, divided_by{sums.data(), total});
  std::vector<double> host = to_host(sums, stream);
  return {std::move(sums), {std::move(host), total, {}}};
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
  int64_t chunk;

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

template <typename T, typename U>
std::vector<double> batched_gram(const equilibration<T>& q,
                                 ritz::gram_batch batch) {
  const rmm::cuda_stream_view stream = stream_of(q.handle);
  const int64_t cells = q.d * q.d;
  buffer<double> gram = to_device(std::vector<double>(cells, 0.0), stream);
  const int64_t rows = batch.blocks * batch.run;
  buffer<U> parts = scratch<U>(batch.blocks * cells, stream);
  cublasHandle_t cublas = cublas_on(q.handle);
  q.template each_block<U>(rows, [&](const U* block, int64_t start) {
    const int64_t count =
        std::min(batch.blocks, (q.n - start + batch.run - 1) / batch.run);
    wide_block_grams<U>(cublas, block, rows, q.d, {batch.run, count},
                        parts.data());
    const buffer<double> sums =
        column_sums(q.handle, block_part<U>{parts.data(), cells}, count, cells);
    each(q.handle, cells, added_into{sums.data(), gram.data()});
  });
  return to_host(gram, stream);
}

template <typename T>
std::vector<double> narrow_gram(const equilibration<T>& q) {
  const rmm::cuda_stream_view stream = stream_of(q.handle);
  const int64_t cells = q.d * q.d;
  buffer<double> gram = to_device(std::vector<double>(cells, 0.0), stream);
  const int64_t total = (q.n + block_rows - 1) / block_rows;
  const int64_t blocks = std::min(total, block_floats / cells);
  buffer<double> parts = scratch<double>(blocks * cells, stream);
  const block_rows_of<T> rows{q.x, q.w, q.mean.data(), q.scales.data(),
                              q.n, q.d};
  for (int64_t first = 0; first < total; first += blocks) {
    const int64_t count = std::min(blocks, total - first);
    narrow_block_grams(stream, rows, first, count, parts.data());
    const buffer<double> sums = column_sums(
        q.handle, block_part<double>{parts.data(), cells}, count, cells);
    each(q.handle, cells, added_into{sums.data(), gram.data()});
  }
  return to_host(gram, stream);
}

template <typename T>
std::vector<double> lower_gram(const equilibration<T>& q) {
  if (q.d <= narrow_gram_columns) {
    return narrow_gram(q);
  }
  return batched_gram<T, float>(q, ritz::wide_batch({q.n, q.d}));
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
  std::vector<double> out;
  out.reserve(d * c);
  buffer<double> z = scratch<double>(q.n * std::min(c, q.chunk), stream);
  for (int64_t first = 0; first < c; first += q.chunk) {
    const int64_t k = std::min(q.chunk, c - first);
    const buffer<double> on_device =
        to_device(std::vector<double>(u.begin() + (first * d),
                                      u.begin() + ((first + k) * d)),
                  stream);
    each(q.handle, q.n * k,
         projected_rows<T>{q.x, q.w, q.mean.data(), on_device.data(), q.n, d,
                           z.data()});
    const std::vector<double> part = to_host(
        column_sums(q.handle,
                    weighted_product<T>{q.x, q.mean.data(), z.data(), q.n, d},
                    q.n, d * k),
        stream);
    out.insert(out.end(), part.begin(), part.end());
  }
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
double exact_condition(const equilibration<T>& q, double error, double limit) {
  const int64_t d = q.d;
  std::vector<double> m =
      batched_gram<T, double>(q, ritz::exact_batch({q.n, d}));
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += q.alpha * q.scale[k] * q.scale[k];
  }
  const std::optional<eigen_pairs> e = eigen_of(q.handle, m, d, true);
  if (!e) {
    return infinite;
  }
  const std::optional<ritz::spectrum_interval> s =
      ritz::enclosure(e->values, enclosure_norms_of(q.handle, m, *e, d), d);
  if (s) {
    if (const auto k =
            ritz::certified_kappa({s->low - error, s->high + error}, limit)) {
      return *k;
    }
  }
  const double estimate =
      e->values.front() > 0 ? e->values.back() / e->values.front() : infinite;
  return std::max(estimate, limit);
}

template <typename T>
std::optional<double> ritz_condition(const equilibration<T>& q,
                                     const std::vector<float>& m,
                                     const eigen_pairs& e,
                                     const gram_plan& plan,
                                     window_errors errors, double limit) {
  const int64_t d = q.d;
  const int64_t window = plan.window;
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
    return std::nullopt;
  }
  const double* y = small->vectors.data();
  const ritz_pair p{small->values.front(), combined(basis, d, y, window),
                    combined(av, d, y, window)};
  const double largest = h.back() / s.back();
  double smallest = p.value;
  if (e.values.front() <= plan.reach) {
    const std::vector<double> v = refined(e, window, p, d);
    const std::vector<double> rv = applied(q, v, 1);
    smallest = std::min(p.value, dot(v.data(), rv.data(), d));
  }
  if (!(smallest > 0)) {
    return infinite;
  }
  if (!(largest / smallest < limit)) {
    return largest / smallest;
  }
  const std::optional<ritz::spectrum_interval> proven =
      window_spectrum(q.handle, m, e, {basis, av, h, window, c}, errors);
  if (!proven) {
    return std::nullopt;
  }
  return ritz::certified_kappa(*proven, limit);
}

}  // namespace

template <typename T>
condition_estimate equilibrated_condition(const raft::handle_t& handle,
                                          const T* x, const T* w, int64_t n,
                                          int64_t d, T alpha,
                                          bool fit_intercept, double limit,
                                          int64_t chunk) {
  require_some("chunk", "vector", chunk);
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto a = static_cast<double>(alpha);
  centred centre = column_means(handle, x, w, n, d, fit_intercept);
  const buffer<double>& mean = centre.mean;
  const std::vector<double> spread = to_host(
      column_sums(handle, weighted_square<T>{x, w, mean.data(), n}, n, d),
      stream);
  if (std::ranges::any_of(spread, [a](double s) { return !(s + a > 0); })) {
    return {infinite, gram_path::none, 0};
  }
  std::vector<double> scale(d);
  std::ranges::transform(spread, scale.begin(),
                         [a](double s) { return 1.0 / std::sqrt(s + a); });
  const buffer<double> scales = to_device(scale, stream);
  const equilibration<T> q{handle, x, w, mean, scales, scale, n, d, a, chunk};
  const std::vector<double> gram = lower_gram(q);
  std::vector<float> m(gram.begin(), gram.end());
  for (int64_t k = 0; k < d; ++k) {
    m[(k * d) + k] += static_cast<float>(a * scale[k] * scale[k]);
  }
  ritz::gram_bound bound = ritz::bound_of(n, spread, scale, a);
  const double plain = ritz::gram_error(bound);
  bound.mean = ritz::mean_error(n, spread, scale, centre.host);
  if (bound.mean * mean_refinement > plain) {
    centre.host.residual = to_host(
        column_sums(handle, weighted_deviation<T>{x, w, mean.data(), n}, n, d),
        stream);
    bound.mean = ritz::mean_error(n, spread, scale, centre.host);
  }
  const double exact = ritz::exact_error(bound, n);
  const std::optional<eigen_pairs> e = eigen_of(handle, m, d, true);
  if (!e) {
    return {exact_condition(q, exact, limit), gram_path::exact, d};
  }
  const double error = ritz::gram_error(bound);
  const gram_plan plan = plan_of(e->values, error, {n, d}, limit);
  if (const auto k =
          enclosed_kappa(handle, m, *e, plan, {n, d}, limit, error)) {
    return {*k,
            plan.path == gram_path::certified ? plan.path : gram_path::enclosed,
            0};
  }
  const gram_plan window = plan.path == gram_path::certified
                               ? plan_of(e->values, error, {n, d}, 0.0)
                               : plan;
  if (window.path == gram_path::ritz) {
    const window_errors errors{error, ritz::applied_error(bound, {n, d}),
                               bound.mean};
    if (const auto k = ritz_condition(q, m, *e, window, errors, limit)) {
      return {*k, window.path, window.window};
    }
  }
  return {exact_condition(q, exact, limit), gram_path::exact, window.window};
}

template condition_estimate equilibrated_condition<float>(const raft::handle_t&,
                                                          const float*,
                                                          const float*, int64_t,
                                                          int64_t, float, bool,
                                                          double, int64_t);
template condition_estimate equilibrated_condition<double>(
    const raft::handle_t&, const double*, const double*, int64_t, int64_t,
    double, bool, double, int64_t);

}  // namespace cumlrkt::glm
