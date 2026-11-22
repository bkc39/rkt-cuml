#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "svm_device.hpp"

namespace cumlrkt::svm {

namespace {

struct bad_code {
  const int32_t* codes;

  __device__ bool operator()(int64_t i) const {
    return codes[i] != 0 && codes[i] != 1;
  }
};

template <typename T>
struct code_value {
  const int32_t* codes;
  T* out;

  __device__ void operator()(int64_t i) const {
    out[i] = static_cast<T>(codes[i]);
  }
};

template <typename T>
struct class_weight {
  const int32_t* codes;
  const T* w;
  int32_t code;

  __device__ double operator()(int64_t i) const {
    return codes[i] == code && w[i] > T{0} ? static_cast<double>(w[i]) : 0.0;
  }
};

template <typename T>
struct penalty {
  const T* w;
  T C;

  __device__ double operator()(int64_t i) const {
    const T c = C * w[i];
    return static_cast<double>(c);
  }
};

template <typename T>
struct squared_norm {
  const T* x;
  int64_t d;

  __device__ double operator()(int64_t i) const {
    double s = 0.0;
    for (int64_t j = 0; j < d; ++j) {
      const double v = static_cast<double>(x[(i * d) + j]);
      s += v * v;
    }
    return s;
  }
};

template <typename T>
struct magnitude {
  const T* x;

  __device__ double operator()(int64_t i) const {
    return fabs(static_cast<double>(x[i]));
  }
};

template <typename T>
struct negative_at {
  const T* x;

  __device__ bool operator()(int64_t i) const {
    return x[i] < T{0};
  }
};

template <typename T>
struct positive_at {
  const T* x;

  __device__ bool operator()(int64_t i) const {
    return x[i] > T{0};
  }
};

template <typename T, typename L>
struct thresholded {
  const T* decision;
  const L* classes;
  L* out;

  __device__ void operator()(int64_t i) const {
    const int32_t at = decision[i] >= T{0} ? 1 : 0;
    out[i] = classes == nullptr ? static_cast<L>(at) : classes[at];
  }
};

template <typename F>
double total(int64_t n, F f, rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), begin, begin + n, f,
                                  0.0, thrust::plus<double>());
}

}  // namespace

int64_t first_bad_code(const int32_t* codes, int64_t n,
                       rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const auto found = thrust::find_if(rmm::exec_policy(stream), begin, begin + n,
                                     bad_code{codes});
  const int64_t i = *found;
  return i < n ? i : -1;
}

template <typename T>
void codes_to(const int32_t* codes, int64_t n, T* out,
              rmm::cuda_stream_view stream) {
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), n,
                     code_value<T>{codes, out});
}

template <typename T>
std::pair<double, double> class_weights(const int32_t* codes, const T* w,
                                        int64_t n,
                                        rmm::cuda_stream_view stream) {
  return {total(n, class_weight<T>{codes, w, 0}, stream),
          total(n, class_weight<T>{codes, w, 1}, stream)};
}

template <typename T>
double penalty_total(const T* w, int64_t n, double C,
                     rmm::cuda_stream_view stream) {
  return total(n, penalty<T>{w, static_cast<T>(C)}, stream);
}

template <typename T>
double largest_squared_norm(const T* x, int64_t n, int64_t d,
                            rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), begin, begin + n,
                                  squared_norm<T>{x, d}, 0.0,
                                  thrust::maximum<double>());
}

template <typename T>
double absolute_total(const T* x, int64_t n, rmm::cuda_stream_view stream) {
  return total(n, magnitude<T>{x}, stream);
}

template <typename T>
std::pair<int64_t, int64_t> sign_counts(const T* dual, int64_t n,
                                        rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const auto policy = rmm::exec_policy(stream);
  return {thrust::count_if(policy, begin, begin + n, negative_at<T>{dual}),
          thrust::count_if(policy, begin, begin + n, positive_at<T>{dual})};
}

template <typename T, typename L>
void threshold_labels(const T* decision, int64_t n, const L* classes, L* out,
                      rmm::cuda_stream_view stream) {
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), n,
                     thresholded<T, L>{decision, classes, out});
}

#define CUMLRKT_SVM_LABELS(T, L)                                            \
  template void threshold_labels<T, L>(const T*, int64_t, const L*, L*,     \
                                       rmm::cuda_stream_view);

#define CUMLRKT_SVM_FLOAT(T)                                                \
  template void codes_to<T>(const int32_t*, int64_t, T*,                    \
                            rmm::cuda_stream_view);                         \
  template std::pair<double, double> class_weights<T>(                      \
      const int32_t*, const T*, int64_t, rmm::cuda_stream_view);            \
  template double penalty_total<T>(const T*, int64_t, double,               \
                                   rmm::cuda_stream_view);                  \
  template double largest_squared_norm<T>(const T*, int64_t, int64_t,       \
                                          rmm::cuda_stream_view);           \
  template double absolute_total<T>(const T*, int64_t,                      \
                                    rmm::cuda_stream_view);                 \
  template std::pair<int64_t, int64_t> sign_counts<T>(const T*, int64_t,    \
                                                      rmm::cuda_stream_view); \
  CUMLRKT_SVM_LABELS(T, int32_t)                                            \
  CUMLRKT_SVM_LABELS(T, int64_t)                                            \
  CUMLRKT_SVM_LABELS(T, float)                                              \
  CUMLRKT_SVM_LABELS(T, double)

CUMLRKT_SVM_FLOAT(float)
CUMLRKT_SVM_FLOAT(double)

}  // namespace cumlrkt::svm
