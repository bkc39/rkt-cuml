#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "cumlrkt/support.hpp"
#include "gamma.hpp"

namespace cumlrkt::svm {

namespace {

template <typename T>
struct widened {
  const T* x;

  __device__ double operator()(int64_t i) const {
    return static_cast<double>(x[i]);
  }
};

template <typename T>
struct squared_deviation {
  const T* x;
  double mean;

  __device__ double operator()(int64_t i) const {
    const double v = static_cast<double>(x[i]) - mean;
    return v * v;
  }
};

template <typename F>
double sum_of(int64_t count, F f, rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), begin,
                                  begin + count, f, 0.0,
                                  thrust::plus<double>());
}

}  // namespace

template <typename T>
T element_variance(const T* x, int64_t count, rmm::cuda_stream_view stream) {
  const auto n = static_cast<double>(count);
  const double mean = sum_of(count, widened<T>{x}, stream) / n;
  return static_cast<T>(sum_of(count, squared_deviation<T>{x, mean}, stream) /
                        n);
}

template <typename T>
double scale_gamma(const T* x, int64_t n, int64_t d,
                   rmm::cuda_stream_view stream) {
  const auto var = static_cast<double>(element_variance(x, n * d, stream));
  if (!(var > 0.0)) {
    throw logic_error(
        "gamma 'scale: X has zero variance; pass a numeric #:gamma");
  }
  return 1.0 / (static_cast<double>(d) * var);
}

template float element_variance<float>(const float*, int64_t,
                                       rmm::cuda_stream_view);
template double element_variance<double>(const double*, int64_t,
                                         rmm::cuda_stream_view);
template double scale_gamma<float>(const float*, int64_t, int64_t,
                                   rmm::cuda_stream_view);
template double scale_gamma<double>(const double*, int64_t, int64_t,
                                    rmm::cuda_stream_view);

}  // namespace cumlrkt::svm
