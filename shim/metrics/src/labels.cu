#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "labels.hpp"

namespace cumlrkt::metrics {

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

template dense_labels<int32_t> dense_codes<int32_t>(const int32_t*, int64_t,
                                                    rmm::cuda_stream_view);
template dense_labels<int64_t> dense_codes<int64_t>(const int64_t*, int64_t,
                                                    rmm::cuda_stream_view);

}  // namespace cumlrkt::metrics
