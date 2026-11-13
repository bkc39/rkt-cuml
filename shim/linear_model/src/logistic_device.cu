#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/support.hpp"
#include "glm_device.cuh"
#include "logistic_device.hpp"

namespace cumlrkt::glm::logistic {

namespace {

template <typename L>
struct non_integral {
  const L* y;

  __device__ bool operator()(int64_t i) const {
    const double v = static_cast<double>(y[i]);
    return !(isfinite(v) && v == floor(v));
  }
};

template <typename T>
struct folded_weight {
  const T* w;
  const int32_t* codes;
  const double* class_weight;
  double* out;

  __device__ void operator()(int64_t i) const {
    const double base = w == nullptr ? 1.0 : static_cast<double>(w[i]);
    out[i] = class_weight == nullptr ? base : base * class_weight[codes[i]];
  }
};

struct class_entry {
  const double* w;
  const int32_t* codes;

  __device__ double operator()(int64_t i, int64_t c) const {
    return codes[i] == c ? (w == nullptr ? 1.0 : w[i]) : 0.0;
  }
};

struct positive_weight {
  const double* w;

  __device__ bool operator()(int64_t i) const {
    return w[i] > 0;
  }
};

struct weight_entry {
  const double* w;

  __device__ double operator()(int64_t i) const {
    return w[i];
  }
};

struct range {
  double least;
  double most;
};

struct weight_range {
  const double* w;

  __device__ range operator()(int64_t i) const {
    return {w[i], w[i]};
  }
};

struct widen {
  __device__ range operator()(range a, range b) const {
    return {a.least < b.least ? a.least : b.least,
            a.most < b.most ? b.most : a.most};
  }
};

struct first_sums {
  double weighted;
  double weight;
  double largest;
  int varies;
};

struct add_first {
  __device__ first_sums operator()(first_sums a, first_sums b) const {
    return {a.weighted + b.weighted, a.weight + b.weight,
            a.largest < b.largest ? b.largest : a.largest, a.varies | b.varies};
  }
};

template <typename T>
struct first_entry {
  const T* x;
  const double* w;
  int64_t d;
  int64_t reference;

  __device__ first_sums operator()(int64_t i, int64_t j) const {
    const double v = static_cast<double>(x[(i * d) + j]);
    const double wi = w == nullptr ? 1.0 : w[i];
    return {wi * v, wi, fabs(v), x[(i * d) + j] != x[(reference * d) + j]};
  }
};

template <typename T>
struct deviation_entry {
  const T* x;
  const double* w;
  const double* mean;
  const int* exponent;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t j) const {
    const double c = scalbn(static_cast<double>(x[(i * d) + j]), -exponent[j]) -
                     scalbn(mean[j], -exponent[j]);
    return (w == nullptr ? 1.0 : w[i]) * c * c;
  }
};

template <typename T>
struct staged_row {
  const T* x;
  row_selection rows;
  const int64_t* kept;
  int64_t p;
  int64_t d;
  int shift;
  T* out;

  __device__ void operator()(int64_t o) const {
    const int64_t r = o / p;
    const int64_t q = o - (r * p);
    out[o] = scaled(x[(source_row(rows, r) * d) + kept[q]], shift);
  }
};

template <typename T>
struct staged_target {
  const int32_t* codes;
  const double* weights;
  double scale;
  row_selection rows;
  T* y;
  T* w;

  __device__ void operator()(int64_t r) const {
    const int64_t i = source_row(rows, r);
    y[r] = static_cast<T>(codes[i]);
    if (w != nullptr) {
      w[r] = static_cast<T>(weights[i] * scale);
    }
  }
};

template <typename T>
struct unstaged_coef {
  const T* params;
  const int64_t* slot;
  int64_t d;
  int64_t k;
  int shift;
  T* coef;

  __device__ void operator()(int64_t o) const {
    const int64_t c = o / d;
    const int64_t j = o - (c * d);
    const int64_t s = slot[j];
    coef[o] = s < 0 ? T{0} : scaled(params[(s * k) + c], shift);
  }
};

template <typename T>
struct packed_params {
  const T* coef;
  const T* intercept;
  int64_t d;
  int64_t k;
  T* params;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / k;
    const int64_t c = o - (j * k);
    params[o] = j < d ? coef[(c * d) + j] : intercept[c];
  }
};

__device__ inline double softplus(double v) {
  return (v > 0 ? v : 0.0) + log1p(exp(-fabs(v)));
}

template <typename T>
struct row_residual {
  const T* x;
  const T* y;
  const T* w;
  const T* params;
  qn_shape s;
  double* r;
  double* loss;

  __device__ double score(int64_t i, int64_t c) const {
    double z =
        s.fit_intercept ? static_cast<double>(params[(s.p * s.k) + c]) : 0.0;
    for (int64_t j = 0; j < s.p; ++j) {
      z += static_cast<double>(x[(i * s.p) + j]) *
           static_cast<double>(params[(j * s.k) + c]);
    }
    return z;
  }

  __device__ void operator()(int64_t i) const {
    const double wi = w == nullptr ? 1.0 : static_cast<double>(w[i]);
    const auto label = static_cast<int64_t>(y[i]);
    if (s.k == 1) {
      const double z = score(i, 0);
      const double t = label == 1 ? 1.0 : -1.0;
      loss[i] = wi * softplus(-t * z);
      r[i] = wi * ((1.0 / (1.0 + exp(-z))) - static_cast<double>(label));
      return;
    }
    double top = -INFINITY;
    for (int64_t c = 0; c < s.k; ++c) {
      r[(i * s.k) + c] = score(i, c);
      top = fmax(top, r[(i * s.k) + c]);
    }
    double total = 0;
    for (int64_t c = 0; c < s.k; ++c) {
      total += exp(r[(i * s.k) + c] - top);
    }
    const double lse = top + log(total);
    loss[i] = wi * (lse - r[(i * s.k) + label]);
    for (int64_t c = 0; c < s.k; ++c) {
      const double prob = exp(r[(i * s.k) + c] - lse);
      r[(i * s.k) + c] = wi * (prob - (c == label ? 1.0 : 0.0));
    }
  }
};

template <typename T>
struct residual_entry {
  const T* x;
  const double* r;
  const double* loss;
  qn_shape s;

  __device__ double operator()(int64_t i, int64_t q) const {
    const int64_t cells = s.p * s.k;
    if (q < cells) {
      const int64_t j = q / s.k;
      return r[(i * s.k) + (q - (j * s.k))] *
             static_cast<double>(x[(i * s.p) + j]);
    }
    if (q < cells + s.k) {
      return r[(i * s.k) + (q - cells)];
    }
    return loss[i];
  }
};

template <typename T>
struct probabilities {
  const T* scores;
  int64_t k;
  int64_t classes;
  bool logarithm;
  T* out;

  __device__ void operator()(int64_t i) const {
    if (k == 1) {
      const double z = static_cast<double>(scores[i]);
      if (logarithm) {
        out[(i * 2) + 0] = static_cast<T>(-softplus(z));
        out[(i * 2) + 1] = static_cast<T>(-softplus(-z));
      } else {
        const auto p1 = static_cast<T>(1.0 / (1.0 + exp(-z)));
        out[(i * 2) + 1] = p1;
        out[(i * 2) + 0] = static_cast<T>(1.0 - static_cast<double>(p1));
      }
      return;
    }
    double top = -INFINITY;
    for (int64_t c = 0; c < classes; ++c) {
      top = fmax(top, static_cast<double>(scores[(i * k) + c]));
    }
    double total = 0;
    for (int64_t c = 0; c < classes; ++c) {
      total += exp(static_cast<double>(scores[(i * k) + c]) - top);
    }
    const double log_total = log(total);
    for (int64_t c = 0; c < classes; ++c) {
      const double shifted = static_cast<double>(scores[(i * k) + c]) - top;
      out[(i * k) + c] = static_cast<T>(logarithm ? shifted - log_total
                                                  : exp(shifted) / total);
    }
  }
};

template <typename T>
__device__ int64_t argmax_of(const T* scores, int64_t k, int64_t i) {
  if (k == 1) {
    return scores[i] > T{0} ? 1 : 0;
  }
  int64_t best = 0;
  for (int64_t c = 1; c < k; ++c) {
    if (scores[(i * k) + c] > scores[(i * k) + best]) {
      best = c;
    }
  }
  return best;
}

template <typename T>
struct argmax_code {
  const T* scores;
  int64_t k;
  int32_t* out;

  __device__ void operator()(int64_t i) const {
    out[i] = static_cast<int32_t>(argmax_of(scores, k, i));
  }
};

template <typename T, typename L>
struct argmax_label {
  const T* scores;
  int64_t k;
  const L* classes;
  L* out;

  __device__ void operator()(int64_t i) const {
    out[i] = classes[argmax_of(scores, k, i)];
  }
};

}  // namespace

template <typename L>
int64_t first_non_integral(const L* y, int64_t n,
                           rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const auto found = thrust::find_if(rmm::exec_policy(stream), begin, begin + n,
                                     non_integral<L>{y});
  const int64_t i = *found;
  return i < n ? i : -1;
}

template <typename T>
buffer<double> effective_weights(const raft::handle_t& handle, const T* w,
                                 const int32_t* codes,
                                 const double* class_weight, int64_t n) {
  buffer<double> out = scratch<double>(n, stream_of(handle));
  each(handle, n, folded_weight<T>{w, codes, class_weight, out.data()});
  return out;
}

std::vector<double> class_totals(rmm::cuda_stream_view stream,
                                 const double* weights, const int32_t* codes,
                                 int64_t n, int64_t k) {
  return column_reduce<column_lanes>(stream, row_selection{nullptr, n}, k,
                                     class_entry{weights, codes},
                                     thrust::plus<double>{}, 0.0);
}

int64_t positive_weights(const double* weights, int64_t n, int64_t* index,
                         rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const auto end = thrust::copy_if(rmm::exec_policy(stream), begin, begin + n,
                                   index, positive_weight{weights});
  return end - index;
}

bool equal_weights(const double* weights, row_selection rows,
                   rmm::cuda_stream_view stream) {
  const range r = vector_reduce(stream, rows, weight_range{weights}, widen{},
                                range{INFINITY, -INFINITY});
  return r.least == r.most;
}

double weight_sum(const double* weights, row_selection rows,
                  rmm::cuda_stream_view stream) {
  return vector_reduce(stream, rows, weight_entry{weights},
                       thrust::plus<double>{}, 0.0);
}

template <typename T>
column_stats column_stats_of(const matrix<T, int64_t>& x, const double* weights,
                             row_selection rows, bool fit_intercept,
                             rmm::cuda_stream_view stream) {
  const int64_t reference =
      rows.index == nullptr ? 0 : to_host(rows.index, 1, stream)[0];
  const std::vector<first_sums> first = column_reduce<column_lanes>(
      stream, rows, x.d, first_entry<T>{x.data, weights, x.d, reference},
      add_first{}, first_sums{0.0, 0.0, 0.0, 0});
  std::vector<double> mean(x.d, 0.0);
  if (fit_intercept) {
    std::ranges::transform(first, mean.begin(), [](const first_sums& f) {
      return f.weighted / f.weight;
    });
  }
  std::vector<int> exponent(x.d, 0);
  std::ranges::transform(first, exponent.begin(), [](const first_sums& f) {
    return f.largest > 0 && std::isfinite(f.largest) ? std::ilogb(f.largest)
                                                     : 0;
  });
  const buffer<double> means = to_device(mean, stream);
  const buffer<int> exponents = to_device(exponent, stream);
  const std::vector<double> square = column_reduce<column_lanes>(
      stream, rows, x.d,
      deviation_entry<T>{x.data, weights, means.data(), exponents.data(), x.d},
      thrust::plus<double>{}, 0.0);
  column_stats s{std::vector<double>(x.d),
                 std::move(mean),
                 std::vector<double>(x.d),
                 {},
                 0.0};
  for (int64_t j = 0; j < x.d; ++j) {
    const first_sums& f = first[j];
    s.sigma[j] = std::ldexp(std::sqrt(square[j] / f.weight), exponent[j]);
    s.magnitude[j] = f.largest;
    s.largest = std::max(s.largest, f.largest);
    const bool spread =
        !(s.sigma[j] <= constant_spread * unit_roundoff<T> * f.largest);
    if ((fit_intercept ? f.varies != 0 : f.largest > 0) && spread) {
      s.kept.push_back(j);
    }
  }
  return s;
}

template <typename T>
void stage_rows(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                row_selection rows, const int64_t* kept, int64_t p, int shift,
                T* out) {
  each(handle, rows.count * p,
       staged_row<T>{x.data, rows, kept, p, x.d, shift, out});
}

template <typename T>
void staged_targets(const raft::handle_t& handle, const int32_t* codes,
                    const double* weights, double scale, row_selection rows,
                    T* y, T* w) {
  each(handle, rows.count, staged_target<T>{codes, weights, scale, rows, y, w});
}

template <typename T>
void params_to_coef(const raft::handle_t& handle, const T* params,
                    const int64_t* slot, int64_t d, int64_t k, int shift,
                    bool fit_intercept, int64_t p, T* coef, T* intercept) {
  each(handle, d * k, unstaged_coef<T>{params, slot, d, k, shift, coef});
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (fit_intercept) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(intercept, params + (p * k), k * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream.value()),
        "cudaMemcpyAsync");
  } else {
    raftrkt::cuda_check(
        cudaMemsetAsync(intercept, 0, k * sizeof(T), stream.value()),
        "cudaMemsetAsync");
  }
}

template <typename T>
void coef_to_params(const raft::handle_t& handle, const T* coef,
                    const T* intercept, int64_t d, int64_t k, T* params) {
  each(handle, (d + 1) * k, packed_params<T>{coef, intercept, d, k, params});
}

template <typename T>
residual_sums residuals_of(const raft::handle_t& handle, const T* x, const T* y,
                           const T* w, const T* params, qn_shape shape) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> r = scratch<double>(shape.m * shape.k, stream);
  buffer<double> loss = scratch<double>(shape.m, stream);
  each(handle, shape.m,
       row_residual<T>{x, y, w, params, shape, r.data(), loss.data()});
  const int64_t cells = shape.p * shape.k;
  const std::vector<double> sums = column_reduce<column_lanes>(
      stream, row_selection{nullptr, shape.m}, cells + shape.k + 1,
      residual_entry<T>{x, r.data(), loss.data(), shape},
      thrust::plus<double>{}, 0.0);
  return {
      std::vector<double>(sums.begin(), sums.begin() + cells),
      std::vector<double>(sums.begin() + cells, sums.begin() + cells + shape.k),
      sums.back()};
}

template <typename T>
void scores_to(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, bool logarithm, T* out) {
  const int64_t classes = k == 1 ? 2 : k;
  each(handle, n, probabilities<T>{scores, k, classes, logarithm, out});
}

template <typename T>
void codes_of(const raft::handle_t& handle, const T* scores, int64_t n,
              int64_t k, int32_t* out) {
  each(handle, n, argmax_code<T>{scores, k, out});
}

template <typename T, typename L>
void labels_of(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, const L* classes, L* out) {
  each(handle, n, argmax_label<T, L>{scores, k, classes, out});
}

#define CUMLRKT_LOGISTIC_FLOAT(T)                                              \
  template buffer<double> effective_weights<T>(const raft::handle_t&,          \
                                               const T*, const int32_t*,       \
                                               const double*, int64_t);        \
  template column_stats column_stats_of<T>(const matrix<T, int64_t>&,          \
                                           const double*, row_selection, bool, \
                                           rmm::cuda_stream_view);             \
  template void stage_rows<T>(const raft::handle_t&,                           \
                              const matrix<T, int64_t>&, row_selection,        \
                              const int64_t*, int64_t, int, T*);               \
  template void staged_targets<T>(const raft::handle_t&, const int32_t*,       \
                                  const double*, double, row_selection, T*,    \
                                  T*);                                         \
  template void params_to_coef<T>(const raft::handle_t&, const T*,             \
                                  const int64_t*, int64_t, int64_t, int, bool, \
                                  int64_t, T*, T*);                            \
  template void coef_to_params<T>(const raft::handle_t&, const T*, const T*,   \
                                  int64_t, int64_t, T*);                       \
  template residual_sums residuals_of<T>(const raft::handle_t&, const T*,      \
                                         const T*, const T*, const T*,         \
                                         qn_shape);                            \
  template void scores_to<T>(const raft::handle_t&, const T*, int64_t,         \
                             int64_t, bool, T*);                               \
  template void codes_of<T>(const raft::handle_t&, const T*, int64_t, int64_t, \
                            int32_t*);

#define CUMLRKT_LOGISTIC_LABEL(L)                                          \
  template int64_t first_non_integral<L>(const L*, int64_t,                \
                                         rmm::cuda_stream_view);           \
  template void labels_of<float, L>(const raft::handle_t&, const float*,   \
                                    int64_t, int64_t, const L*, L*);       \
  template void labels_of<double, L>(const raft::handle_t&, const double*, \
                                     int64_t, int64_t, const L*, L*);

CUMLRKT_LOGISTIC_FLOAT(float)
CUMLRKT_LOGISTIC_FLOAT(double)
CUMLRKT_LOGISTIC_LABEL(int32_t)
CUMLRKT_LOGISTIC_LABEL(int64_t)
CUMLRKT_LOGISTIC_LABEL(float)
CUMLRKT_LOGISTIC_LABEL(double)

}  // namespace cumlrkt::glm::logistic
