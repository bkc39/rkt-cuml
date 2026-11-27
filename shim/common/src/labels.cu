#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "cumlrkt/labels.hpp"

namespace cumlrkt {

namespace {

template <typename T>
struct encoded {
  const T* labels;
  const T* classes;
  int64_t k;
  int32_t* codes;

  __device__ void operator()(int64_t i) const {
    const T v = labels[i];
    int64_t lo = 0;
    int64_t hi = k;
    while (lo < hi) {
      const int64_t mid = (lo + hi) / 2;
      if (classes[mid] < v) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    codes[i] = lo < k && classes[lo] == v ? static_cast<int32_t>(lo) : -1;
  }
};

template <typename T>
struct non_integral {
  const T* labels;

  __device__ bool operator()(int64_t i) const {
    const double v = static_cast<double>(labels[i]);
    return !(isfinite(v) && v == floor(v));
  }
};

}  // namespace

template <typename T>
dense_labels<T> dense_codes(const T* labels, int64_t n,
                            rmm::cuda_stream_view stream) {
  const auto size = static_cast<std::size_t>(n);
  rmm::device_uvector<T> values(size, stream);
  rmm::device_uvector<int32_t> codes(size, stream);
  if (n == 0) {
    return {std::move(codes), std::move(values), 0};
  }
  const auto policy = rmm::exec_policy(stream);
  thrust::copy(policy, labels, labels + n, values.begin());
  thrust::sort(policy, values.begin(), values.end());
  const auto last = thrust::unique(policy, values.begin(), values.end());
  const int64_t k = last - values.begin();
  thrust::lower_bound(policy, values.begin(), last, labels, labels + n,
                      codes.begin());
  values.resize(static_cast<std::size_t>(k), stream);
  values.shrink_to_fit(stream);
  return {std::move(codes), std::move(values), k};
}

template <typename T>
void encode_against(const T* labels, int64_t n, const T* classes, int64_t k,
                    int32_t* codes, rmm::cuda_stream_view stream) {
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), n,
                     encoded<T>{labels, classes, k, codes});
}

template <typename T>
int64_t first_non_integral(const T* labels, int64_t n,
                           rmm::cuda_stream_view stream) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const auto found = thrust::find_if(rmm::exec_policy(stream), begin, begin + n,
                                     non_integral<T>{labels});
  const int64_t i = *found;
  return i < n ? i : -1;
}

#define CUMLRKT_LABELS(T)                                               \
  template dense_labels<T> dense_codes<T>(const T*, int64_t,            \
                                          rmm::cuda_stream_view);       \
  template void encode_against<T>(const T*, int64_t, const T*, int64_t, \
                                  int32_t*, rmm::cuda_stream_view);     \
  template int64_t first_non_integral<T>(const T*, int64_t,             \
                                         rmm::cuda_stream_view);

CUMLRKT_LABELS(int32_t)
CUMLRKT_LABELS(int64_t)
CUMLRKT_LABELS(float)
CUMLRKT_LABELS(double)

}  // namespace cumlrkt
