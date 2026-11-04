#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_device.cuh"
#include "glm_staging_device.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
struct staged_magnitude {
  const T* x;
  const T* w;
  staged_columns columns;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    return fabs(static_cast<double>(
        scaled(x[(i * d) + columns.kept[k]], columns.exponent[k])));
  }
};

template <typename T>
struct normalised_column {
  const T* x;
  staged_columns columns;
  const int* norm;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t k) const {
    return scalbn(static_cast<double>(scaled(x[(i * d) + columns.kept[k]],
                                             columns.exponent[k])),
                  norm[k]);
  }
};

template <typename T>
struct column_weighted {
  normalised_column<T> v;
  const T* w;

  __device__ double operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    return weight(w, i) * v(i, k);
  }
};

template <typename T>
struct centred_residual {
  normalised_column<T> v;
  const T* y;
  const T* coef;
  const double* mean;
  double y_mean;
  double* r;

  __device__ void operator()(int64_t i) const {
    double fitted = 0;
    for (int64_t k = 0; k < v.columns.count; ++k) {
      fitted += (v(i, k) - mean[k]) *
                scalbn(static_cast<double>(coef[k]), -v.norm[k]);
    }
    r[i] = (static_cast<double>(y[i]) - y_mean) - fitted;
  }
};

struct exponent_pair {
  int square;
  int cross;
};

struct larger_pair {
  __device__ exponent_pair operator()(exponent_pair a, exponent_pair b) const {
    return {a.square < b.square ? b.square : a.square,
            a.cross < b.cross ? b.cross : a.cross};
  }
};

struct sum_pair {
  double square;
  double cross;
};

struct add_pairs {
  __device__ sum_pair operator()(sum_pair a, sum_pair b) const {
    return {a.square + b.square, a.cross + b.cross};
  }
};

template <typename T>
struct moment_terms {
  normalised_column<T> v;
  const T* w;
  const double* mean;
  const double* r;

  __device__ product3 square(int64_t i, int64_t k) const {
    const double c = v(i, k) - mean[k];
    return product_of(weight(w, i), c, c);
  }

  __device__ product3 cross(int64_t i, int64_t k) const {
    return product_of(weight(w, i), v(i, k) - mean[k], r[i]);
  }
};

template <typename T>
struct moment_exponents {
  moment_terms<T> t;

  __device__ exponent_pair operator()(int64_t i, int64_t k) const {
    if (!counted(t.w, i)) {
      return {no_exponent, no_exponent};
    }
    return {t.square(i, k).exponent, t.cross(i, k).exponent};
  }
};

template <typename T>
struct scaled_moments {
  moment_terms<T> t;
  const exponent_pair* largest;

  __device__ sum_pair operator()(int64_t i, int64_t k) const {
    if (!counted(t.w, i)) {
      return {0.0, 0.0};
    }
    return {scaled_product(t.square(i, k), largest[k].square),
            scaled_product(t.cross(i, k), largest[k].cross)};
  }
};

template <typename T>
struct y_deviation {
  const T* y;
  const T* w;
  double mean;

  __device__ product3 operator()(int64_t i) const {
    if (!counted(w, i)) {
      return {0.0, no_exponent};
    }
    const double c = static_cast<double>(y[i]) - mean;
    return product_of(weight(w, i), c, c);
  }
};

template <typename T>
struct y_exponent_of {
  y_deviation<T> f;

  __device__ int operator()(int64_t i) const {
    return f(i).exponent;
  }
};

template <typename T>
struct y_scaled {
  y_deviation<T> f;
  int largest;

  __device__ double operator()(int64_t i) const {
    return scaled_product(f(i), largest);
  }
};

scaled_sum sum_of(double value, int exponent) {
  return {value, value == 0 ? 0 : exponent};
}

template <typename T>
scaled_sum y_square(rmm::cuda_stream_view stream, int64_t n, y_deviation<T> f) {
  const thrust::counting_iterator<int64_t> i(0);
  const int e = thrust::transform_reduce(rmm::exec_policy(stream), i, i + n,
                                         y_exponent_of<T>{f}, no_exponent,
                                         largest_exponent{});
  return sum_of(total(stream, n, y_scaled<T>{f, e}), e);
}

}  // namespace

template <typename T>
residual_terms residual_moments(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, staged_columns columns,
                                const T* coef, bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t d = columns.count;
  std::vector<double> largest = column_totals<double>(
      stream, x.n, d, staged_magnitude<T>{x.data, w, columns, x.d}, larger{});
  std::vector<int> norm(largest.size());
  std::ranges::transform(largest, norm.begin(),
                         [](double m) { return unit_exponent(m); });
  const buffer<int> norms = to_device(norm, stream);
  const normalised_column<T> v{x.data, columns, norms.data(), x.d};
  std::vector<double> mean(d, 0.0);
  double y_mean = 0;
  if (fit_intercept) {
    const double weights = total(stream, x.n, weight_of<T>{w});
    const std::vector<double> sums = column_totals<double>(
        stream, x.n, d, column_weighted<T>{v, w}, thrust::plus<double>{});
    std::ranges::transform(sums, mean.begin(),
                           [&](double s) { return s / weights; });
    y_mean = total(stream, x.n, weighted_value<T>{y, w}) / weights;
  }
  const buffer<double> means = to_device(mean, stream);
  buffer<double> r = scratch<double>(x.n, stream);
  each(handle, x.n,
       centred_residual<T>{v, y, coef, means.data(), y_mean, r.data()});
  const moment_terms<T> t{v, w, means.data(), r.data()};
  const std::vector<exponent_pair> top = column_totals<exponent_pair>(
      stream, x.n, d, moment_exponents<T>{t}, larger_pair{});
  const buffer<exponent_pair> tops = to_device(top, stream);
  const std::vector<sum_pair> sums = column_totals<sum_pair>(
      stream, x.n, d, scaled_moments<T>{t, tops.data()}, add_pairs{});
  std::vector<moment> moments(sums.size());
  for (size_t k = 0; k < sums.size(); ++k) {
    moments[k] = {sum_of(sums[k].square, top[k].square),
                  sum_of(sums[k].cross, top[k].cross)};
  }
  return {std::move(moments), std::move(largest),
          y_square(stream, x.n, y_deviation<T>{y, w, y_mean})};
}

template residual_terms residual_moments<float>(const raft::handle_t&,
                                                const matrix<float, int64_t>&,
                                                const float*, const float*,
                                                staged_columns, const float*,
                                                bool);
template residual_terms residual_moments<double>(const raft::handle_t&,
                                                 const matrix<double, int64_t>&,
                                                 const double*, const double*,
                                                 staged_columns, const double*,
                                                 bool);

}  // namespace cumlrkt::glm
