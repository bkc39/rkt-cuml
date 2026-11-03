#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <string>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"

namespace cumlrkt::glm {

namespace {

rmm::cuda_stream_view stream_of(const raft::handle_t& handle) {
  return raft::resource::get_cuda_stream(handle);
}

template <typename T>
std::vector<T> to_host(const buffer<T>& b, rmm::cuda_stream_view stream) {
  std::vector<T> host(b.size());
  raftrkt::cuda_check(
      cudaMemcpyAsync(host.data(), b.data(), b.size() * sizeof(T),
                      cudaMemcpyDeviceToHost, stream.value()),
      "cudaMemcpyAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
  return host;
}

template <typename T>
buffer<T> to_device(const std::vector<T>& host, rmm::cuda_stream_view stream) {
  buffer<T> b = scratch<T>(static_cast<int64_t>(host.size()), stream);
  if (!host.empty()) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(b.data(), host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream.value()),
        "cudaMemcpyAsync");
  }
  return b;
}

template <typename T>
__device__ bool counted(const T* w, int64_t i) {
  return w == nullptr || w[i] > 0;
}

template <typename T>
__device__ double weight(const T* w, int64_t i) {
  return w == nullptr ? 1.0 : static_cast<double>(w[i]);
}

struct column_of {
  int64_t n;

  __device__ int64_t operator()(int64_t k) const {
    return k / n;
  }
};

template <typename F, typename V>
struct by_column {
  F f;
  int64_t n;

  __device__ V operator()(int64_t k) const {
    const int64_t j = k / n;
    return f(k - (j * n), j);
  }
};

template <typename V, typename F, typename Op>
std::vector<V> column_totals(rmm::cuda_stream_view stream, int64_t n,
                             int64_t d, F f, Op op) {
  buffer<V> totals = scratch<V>(d, stream);
  buffer<int64_t> columns = scratch<int64_t>(d, stream);
  const thrust::counting_iterator<int64_t> k(0);
  thrust::reduce_by_key(
      rmm::exec_policy(stream), thrust::make_transform_iterator(k, column_of{n}),
      thrust::make_transform_iterator(k + (n * d), column_of{n}),
      thrust::make_transform_iterator(k, by_column<F, V>{f, n}),
      columns.data(), totals.data(),
      thrust::equal_to<int64_t>{}, op);
  return to_host(totals, stream);
}

template <typename T>
struct positive_magnitude {
  const T* x;
  const T* w;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t j) const {
    return counted(w, i) ? fabs(static_cast<double>(x[(i * d) + j])) : 0.0;
  }
};

struct larger {
  __device__ double operator()(double a, double b) const {
    return a < b ? b : a;
  }
};

template <typename T>
struct vector_magnitude {
  const T* v;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return counted(w, i) ? fabs(static_cast<double>(v[i])) : 0.0;
  }
};

template <typename T>
double largest_counted(const T* v, int64_t n, const T* w,
                       rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> i(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), i, i + n,
                                  vector_magnitude<T>{v, w}, 0.0, larger{});
}

template <typename T>
struct unlike_reference {
  const T* x;
  const T* w;
  int64_t d;
  int64_t reference;
  int* flags;

  __device__ void operator()(int64_t k) const {
    const int64_t i = k / d;
    const int64_t j = k - (i * d);
    const T base = reference < 0 ? T{0} : x[(reference * d) + j];
    if (counted(w, i) && x[k] != base) {
      flags[j] = 1;
    }
  }
};

template <typename T>
struct positive {
  const T* w;

  __device__ bool operator()(int64_t i) const {
    return w[i] > 0;
  }
};

template <typename T>
struct staged_column {
  const T* x;
  const T* w;
  const int64_t* kept;
  const int* e;
  T* out;
  int64_t n;
  int64_t d;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / n;
    const int64_t i = o - (k * n);
    const int64_t j = kept[k];
    out[o] = counted(w, i) ? scaled(x[(i * d) + j], e[k]) : T{0};
  }
};

template <typename T>
struct staged_element {
  const T* v;
  const T* w;
  T* out;
  int exponent;

  __device__ void operator()(int64_t i) const {
    out[i] = counted(w, i) ? scaled(v[i], exponent) : T{0};
  }
};

template <typename T>
struct weighted_value {
  const T* y;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i) * static_cast<double>(y[i]);
  }
};

template <typename T>
struct weight_of {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i);
  }
};

template <typename T>
double total(rmm::cuda_stream_view stream, int64_t n, T f) {
  const thrust::counting_iterator<int64_t> i(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), i, i + n, f, 0.0,
                                  thrust::plus<double>{});
}

template <typename T>
struct unstaged {
  const T* staged;
  const int64_t* slot;
  const int* shift;
  T* coef;

  __device__ void operator()(int64_t j) const {
    const int64_t k = slot[j];
    coef[j] = k < 0 ? T{0} : scaled(staged[k], shift[k]);
  }
};

template <typename T>
struct residual_of {
  const T* x;
  const T* y;
  const int64_t* kept;
  const int* e;
  const T* coef;
  double intercept;
  int64_t d;
  int64_t kept_count;
  double* r;

  __device__ void operator()(int64_t i) const {
    double fitted = intercept;
    for (int64_t k = 0; k < kept_count; ++k) {
      const T v = scaled(x[(i * d) + kept[k]], e[k]);
      fitted += static_cast<double>(v) * static_cast<double>(coef[k]);
    }
    r[i] = static_cast<double>(y[i]) - fitted;
  }
};

template <typename T>
struct column_weighted {
  const T* x;
  const T* w;
  const int64_t* kept;
  const int* e;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    return weight(w, i) *
           static_cast<double>(scaled(x[(i * d) + kept[k]], e[k]));
  }
};

struct pair_sum {
  double first;
  double second;
};

struct add_pairs {
  __device__ pair_sum operator()(pair_sum a, pair_sum b) const {
    return {a.first + b.first, a.second + b.second};
  }
};

template <typename T>
struct column_moments {
  const T* x;
  const T* w;
  const int64_t* kept;
  const int* e;
  const double* mean;
  const double* r;
  int64_t d;

  __device__ pair_sum operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    const double v =
        static_cast<double>(scaled(x[(i * d) + kept[k]], e[k])) - mean[k];
    const double wv = weight(w, i) * v;
    return {wv * v, wv * r[i]};
  }
};

template <typename T>
struct centred_square {
  const T* y;
  const T* w;
  double mean;

  __device__ double operator()(int64_t i) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    const double v = static_cast<double>(y[i]) - mean;
    return weight(w, i) * v * v;
  }
};

}  // namespace

int even_floor(int x) {
  return x - (((x % 2) + 2) % 2);
}

std::string algorithm_name(int algorithm) {
  switch (algorithm) {
    case CUML_LINEAR_MODEL_OLS_SVD:
      return "svd";
    case CUML_LINEAR_MODEL_OLS_EIG:
      return "eig";
    case CUML_LINEAR_MODEL_OLS_QR:
      return "qr";
    default:
      return "svd-qr";
  }
}

int64_t ols_workspace(int algorithm, int64_t n, int64_t d,
                      int64_t cusolver_lwork) {
  const int64_t m = std::min(n, d);
  switch (algorithm) {
    case CUML_LINEAR_MODEL_OLS_EIG:
      return (3 * d * d) + (2 * d);
    case CUML_LINEAR_MODEL_OLS_SVD:
      return cusolver_lwork + (n * m) + (d * m) + (2 * m) + 1;
    case CUML_LINEAR_MODEL_OLS_SVD_QR:
      return cusolver_lwork + (n * m) + (d * d) + (2 * m) + 1;
    default:
      return cusolver_lwork;
  }
}

template <typename T>
int64_t first_positive(const T* w, int64_t n, rmm::cuda_stream_view stream) {
  if (w == nullptr) {
    return 0;
  }
  const thrust::counting_iterator<int64_t> i(0);
  const int64_t found = *thrust::find_if(rmm::exec_policy(stream), i, i + n,
                                         positive<T>{w});
  return found == n ? -1 : found;
}

template <typename T>
std::vector<char> constant_columns(const raft::handle_t& handle,
                                   const matrix<T, int64_t>& x, const T* w,
                                   bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t reference =
      fit_intercept ? first_positive(w, x.n, stream) : int64_t{-1};
  buffer<int> flags = scratch<int>(x.d, stream);
  raftrkt::cuda_check(cudaMemsetAsync(flags.data(), 0, flags.size() * sizeof(int),
                                      stream.value()),
                      "cudaMemsetAsync");
  each(handle, x.n * x.d,
       unlike_reference<T>{x.data, w, x.d, reference, flags.data()});
  const std::vector<int> varies = to_host(flags, stream);
  std::vector<char> constant(varies.size());
  std::ranges::transform(varies, constant.begin(),
                         [](int v) { return static_cast<char>(v == 0); });
  return constant;
}

template <typename T>
bool constant_vector(const raft::handle_t& handle, const T* v, int64_t n,
                     const T* w, bool fit_intercept) {
  const std::vector<char> c =
      constant_columns(handle, matrix<T, int64_t>{const_cast<T*>(v), n, 1}, w,
                       fit_intercept);
  return c[0] != 0;
}

template <typename T>
staging plan_staging(const raft::handle_t& handle,
                     const matrix<T, int64_t>& x, const T* y, const T* w,
                     bool fit_intercept, exponents mode) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const std::vector<double> largest = column_totals<double>(
      stream, x.n, x.d, positive_magnitude<T>{x.data, w, x.d}, larger{});
  const std::vector<char> constant =
      constant_columns(handle, x, w, fit_intercept);
  staging s;
  double widest = 0;
  for (int64_t j = 0; j < x.d; ++j) {
    if (constant[j] == 0) {
      s.kept.push_back(j);
      widest = std::max(widest, largest[j]);
    }
  }
  s.e.resize(x.d);
  std::ranges::transform(largest, s.e.begin(), [&](double m) {
    return unit_exponent(mode == exponents::uniform ? widest : m);
  });
  s.f = unit_exponent(largest_counted(y, x.n, w, stream));
  s.g = w == nullptr
            ? 0
            : even_floor(unit_exponent(largest_counted(w, x.n, w, stream)));
  return s;
}

template <typename T>
void stage_columns(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                   const T* w, const staging& s, T* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  std::vector<int> e(s.kept.size());
  std::ranges::transform(s.kept, e.begin(), [&](int64_t j) { return s.e[j]; });
  const buffer<int64_t> kept = to_device(s.kept, stream);
  const buffer<int> exponents = to_device(e, stream);
  const auto count = static_cast<int64_t>(s.kept.size());
  each(handle, x.n * count,
       staged_column<T>{x.data, w, kept.data(), exponents.data(), out, x.n,
                        x.d});
  settle(handle);
}

template <typename T>
void stage_vector(const raft::handle_t& handle, const T* v, int64_t n,
                  const T* w, int exponent, T* out) {
  each(handle, n, staged_element<T>{v, w, out, exponent});
}

template <typename T>
double staged_mean(const raft::handle_t& handle, const T* y, const T* w,
                   int64_t n) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  return total(stream, n, weighted_value<T>{y, w}) /
         total(stream, n, weight_of<T>{w});
}

template <typename T>
int64_t unstage_coef(const raft::handle_t& handle, const T* staged,
                     const staging& s, int64_t d, T* coef) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  std::vector<int64_t> slot(d, -1);
  std::vector<int> shift(s.kept.size());
  for (size_t k = 0; k < s.kept.size(); ++k) {
    slot[s.kept[k]] = static_cast<int64_t>(k);
    shift[k] = s.e[s.kept[k]] - s.f;
  }
  const buffer<int64_t> slots = to_device(slot, stream);
  const buffer<int> shifts = to_device(shift, stream);
  each(handle, d, unstaged<T>{staged, slots.data(), shifts.data(), coef});
  const int64_t j = first_non_finite(coef, d, stream);
  settle(handle);
  return j;
}

template <typename T>
void require_finite_solution(const T* staged, int64_t d, T intercept,
                             int algorithm, rmm::cuda_stream_view stream) {
  if (first_non_finite(staged, d, stream) >= 0 || !std::isfinite(intercept)) {
    throw logic_error(std::format("X: the {} solve gave a non-finite coefficient",
                                  algorithm_name(algorithm)));
  }
}

template <typename T>
double residual_cosine(const raft::handle_t& handle,
                       const matrix<T, int64_t>& x, const staging& s,
                       const T* y, const T* w, const T* coef, T intercept,
                       bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto count = static_cast<int64_t>(s.kept.size());
  std::vector<int> e(s.kept.size());
  std::ranges::transform(s.kept, e.begin(), [&](int64_t j) { return s.e[j]; });
  const buffer<int64_t> kept = to_device(s.kept, stream);
  const buffer<int> exponents = to_device(e, stream);
  buffer<double> r = scratch<double>(x.n, stream);
  each(handle, x.n,
       residual_of<T>{x.data, y, kept.data(), exponents.data(), coef,
                      static_cast<double>(intercept), x.d, count, r.data()});
  const double weights = total(stream, x.n, weight_of<T>{w});
  std::vector<double> mean(count, 0.0);
  double y_mean = 0;
  if (fit_intercept) {
    const std::vector<double> sums = column_totals<double>(
        stream, x.n, count,
        column_weighted<T>{x.data, w, kept.data(), exponents.data(), x.d},
        thrust::plus<double>{});
    std::ranges::transform(sums, mean.begin(),
                           [&](double v) { return v / weights; });
    y_mean = total(stream, x.n, weighted_value<T>{y, w}) / weights;
  }
  const buffer<double> means = to_device(mean, stream);
  const std::vector<pair_sum> moments = column_totals<pair_sum>(
      stream, x.n, count,
      column_moments<T>{x.data, w, kept.data(), exponents.data(), means.data(),
                        r.data(), x.d},
      add_pairs{});
  const double y_norm =
      std::sqrt(total(stream, x.n, centred_square<T>{y, w, y_mean}));
  double worst = 0;
  for (const pair_sum& m : moments) {
    const double g = std::fabs(m.second) / (std::sqrt(m.first) * y_norm);
    worst = std::isnan(g) || std::isnan(worst) ? NAN : std::max(worst, g);
  }
  return worst;
}

template int64_t first_positive<float>(const float*, int64_t,
                                       rmm::cuda_stream_view);
template int64_t first_positive<double>(const double*, int64_t,
                                        rmm::cuda_stream_view);
template std::vector<char> constant_columns<float>(
    const raft::handle_t&, const matrix<float, int64_t>&, const float*, bool);
template std::vector<char> constant_columns<double>(
    const raft::handle_t&, const matrix<double, int64_t>&, const double*,
    bool);
template bool constant_vector<float>(const raft::handle_t&, const float*,
                                     int64_t, const float*, bool);
template bool constant_vector<double>(const raft::handle_t&, const double*,
                                      int64_t, const double*, bool);
template staging plan_staging<float>(const raft::handle_t&,
                                     const matrix<float, int64_t>&,
                                     const float*, const float*, bool,
                                     exponents);
template staging plan_staging<double>(const raft::handle_t&,
                                      const matrix<double, int64_t>&,
                                      const double*, const double*, bool,
                                      exponents);
template void stage_columns<float>(const raft::handle_t&,
                                   const matrix<float, int64_t>&, const float*,
                                   const staging&, float*);
template void stage_columns<double>(const raft::handle_t&,
                                    const matrix<double, int64_t>&,
                                    const double*, const staging&, double*);
template void stage_vector<float>(const raft::handle_t&, const float*, int64_t,
                                  const float*, int, float*);
template void stage_vector<double>(const raft::handle_t&, const double*,
                                   int64_t, const double*, int, double*);
template double staged_mean<float>(const raft::handle_t&, const float*,
                                   const float*, int64_t);
template double staged_mean<double>(const raft::handle_t&, const double*,
                                    const double*, int64_t);
template int64_t unstage_coef<float>(const raft::handle_t&, const float*,
                                     const staging&, int64_t, float*);
template int64_t unstage_coef<double>(const raft::handle_t&, const double*,
                                      const staging&, int64_t, double*);
template void require_finite_solution<float>(const float*, int64_t, float, int,
                                             rmm::cuda_stream_view);
template void require_finite_solution<double>(const double*, int64_t, double,
                                              int, rmm::cuda_stream_view);
template double residual_cosine<float>(const raft::handle_t&,
                                       const matrix<float, int64_t>&,
                                       const staging&, const float*,
                                       const float*, const float*, float, bool);
template double residual_cosine<double>(const raft::handle_t&,
                                        const matrix<double, int64_t>&,
                                        const staging&, const double*,
                                        const double*, const double*, double,
                                        bool);

}  // namespace cumlrkt::glm
