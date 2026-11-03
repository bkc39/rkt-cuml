#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "scores_device.hpp"

namespace cumlrkt::metrics {

namespace {

struct weight_at {
  const float* single;
  const double* wide;
  int exponent;

  __device__ double raw(int64_t i) const {
    if (single != nullptr) {
      return static_cast<double>(single[i]);
    }
    return wide != nullptr ? wide[i] : 1.0;
  }

  __device__ bool positive(int64_t i) const {
    return raw(i) > 0;
  }

  __device__ double operator()(int64_t i) const {
    return scalbn(raw(i), -exponent);
  }
};

template <typename T>
struct value_at {
  const T* x;
  int exponent;

  __device__ double operator()(int64_t i) const {
    return scalbn(static_cast<double>(x[i]), -exponent);
  }
};

__device__ double squared(double v) {
  return __dmul_rn(v, v);
}

template <typename T>
struct weighted_value {
  weight_at w;
  value_at<T> y;

  __device__ double operator()(int64_t i) const {
    return __dmul_rn(w(i), y(i));
  }
};

template <typename T>
struct weighted_deviation {
  weight_at w;
  value_at<T> y;
  double mean;

  __device__ double operator()(int64_t i) const {
    return __dmul_rn(w(i), squared(__dsub_rn(y(i), mean)));
  }
};

template <typename T, typename P>
struct weighted_residual {
  weight_at w;
  value_at<T> y;
  value_at<P> p;
  int64_t d;

  __device__ double operator()(int64_t k) const {
    return __dmul_rn(w(k / d), squared(__dsub_rn(y(k), p(k))));
  }
};

template <typename T>
struct unlike_first {
  weight_at w;
  const T* y;
  int64_t first;

  __device__ bool operator()(int64_t i) const {
    return w.positive(i) && y[i] != y[first];
  }
};

template <typename T, typename P>
struct missed {
  weight_at w;
  const T* y;
  const P* p;

  __device__ bool operator()(int64_t i) const {
    return w.positive(i) &&
           static_cast<double>(y[i]) != static_cast<double>(p[i]);
  }
};

struct positive_weight {
  weight_at w;

  __device__ bool operator()(int64_t i) const {
    return w.positive(i);
  }
};

template <typename A, typename B>
struct matched_weight {
  weight_at w;
  const A* a;
  const B* b;

  __device__ double operator()(int64_t i) const {
    return static_cast<int64_t>(a[i]) == static_cast<int64_t>(b[i]) ? w(i)
                                                                    : 0.0;
  }
};

weight_at weight_of(sample_weights w) {
  return {w.single, w.wide, w.exponent};
}

template <typename F>
double total(F f, int64_t count, rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> indices(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), indices,
                                  indices + count, f, 0.0,
                                  thrust::plus<double>{});
}

template <typename Predicate>
int64_t first_where(Predicate p, int64_t n, rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> indices(0);
  const int64_t i =
      *thrust::find_if(rmm::exec_policy(stream), indices, indices + n, p);
  return i == n ? -1 : i;
}

template <typename Predicate>
bool any_where(Predicate p, int64_t n, rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> indices(0);
  return thrust::any_of(rmm::exec_policy(stream), indices, indices + n, p);
}

}  // namespace

template <typename T, typename P>
double r2_of(const T* y, const P* p, int64_t n, int exponent, sample_weights w,
             rmm::cuda_stream_view stream) {
  const weight_at weight = weight_of(w);
  const int64_t first = first_where(positive_weight{weight}, n, stream);
  if (!any_where(unlike_first<T>{weight, y, first}, n, stream)) {
    return any_where(missed<T, P>{weight, y, p}, n, stream) ? 0.0 : 1.0;
  }
  const value_at<T> truth{y, exponent};
  const value_at<P> pred{p, exponent};
  const double sum = total(weight, n, stream);
  const double mean = total(weighted_value<T>{weight, truth}, n, stream) / sum;
  const double spread =
      total(weighted_deviation<T>{weight, truth, mean}, n, stream);
  const double residual =
      total(weighted_residual<T, P>{weight, truth, pred, 1}, n, stream);
  return residual == 0 ? 1.0 : 1.0 - (residual / spread);
}

template <typename T, typename P>
double squared_error_of(const T* y, const P* p, int64_t n, int64_t d,
                        int exponent, sample_weights w,
                        rmm::cuda_stream_view stream) {
  const weight_at weight = weight_of(w);
  const double sum = total(weight, n, stream);
  const double residual =
      total(weighted_residual<T, P>{weight, value_at<T>{y, exponent},
                                    value_at<P>{p, exponent}, d},
            n * d, stream);
  return std::ldexp(residual / (static_cast<double>(d) * sum), 2 * exponent);
}

template <typename A, typename B>
matched matches_of(const A* a, const B* b, int64_t n, sample_weights w,
                   rmm::cuda_stream_view stream) {
  const weight_at weight = weight_of(w);
  return {total(matched_weight<A, B>{weight, a, b}, n, stream),
          total(weight, n, stream)};
}

template double r2_of<float, float>(const float*, const float*, int64_t, int,
                                    sample_weights, rmm::cuda_stream_view);
template double r2_of<float, double>(const float*, const double*, int64_t, int,
                                     sample_weights, rmm::cuda_stream_view);
template double r2_of<double, float>(const double*, const float*, int64_t, int,
                                     sample_weights, rmm::cuda_stream_view);
template double r2_of<double, double>(const double*, const double*, int64_t,
                                      int, sample_weights,
                                      rmm::cuda_stream_view);
template double squared_error_of<float, float>(const float*, const float*,
                                               int64_t, int64_t, int,
                                               sample_weights,
                                               rmm::cuda_stream_view);
template double squared_error_of<float, double>(const float*, const double*,
                                                int64_t, int64_t, int,
                                                sample_weights,
                                                rmm::cuda_stream_view);
template double squared_error_of<double, float>(const double*, const float*,
                                                int64_t, int64_t, int,
                                                sample_weights,
                                                rmm::cuda_stream_view);
template double squared_error_of<double, double>(const double*, const double*,
                                                 int64_t, int64_t, int,
                                                 sample_weights,
                                                 rmm::cuda_stream_view);
template matched matches_of<int32_t, int32_t>(const int32_t*, const int32_t*,
                                              int64_t, sample_weights,
                                              rmm::cuda_stream_view);
template matched matches_of<int32_t, int64_t>(const int32_t*, const int64_t*,
                                              int64_t, sample_weights,
                                              rmm::cuda_stream_view);
template matched matches_of<int64_t, int32_t>(const int64_t*, const int32_t*,
                                              int64_t, sample_weights,
                                              rmm::cuda_stream_view);
template matched matches_of<int64_t, int64_t>(const int64_t*, const int64_t*,
                                              int64_t, sample_weights,
                                              rmm::cuda_stream_view);

}  // namespace cumlrkt::metrics
