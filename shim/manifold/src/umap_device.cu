#include <thrust/binary_search.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <utility>

#include "cumlrkt/labels.hpp"
#include "umap_device.hpp"

namespace cumlrkt::manifold {

namespace {

constexpr int32_t unlabelled_code = -1;

template <typename T>
struct shifted {
  int64_t unlabelled;

  __device__ float operator()(T label, int32_t code) const {
    if (label == T{-1}) {
      return static_cast<float>(unlabelled_code);
    }
    return static_cast<float>(code - (unlabelled >= 0 && code > unlabelled));
  }
};

}  // namespace

template <typename T>
label_codes float_codes(const T* labels, int64_t n,
                        rmm::cuda_stream_view stream) {
  dense_labels<T> found = dense_codes(labels, n, stream);
  const auto policy = rmm::exec_policy(stream);
  const T* first = found.values.data();
  const T* last = first + found.k;
  const bool has_unlabelled =
      thrust::binary_search(policy, first, last, T{-1});
  const int64_t unlabelled =
      has_unlabelled ? thrust::lower_bound(policy, first, last, T{-1}) - first
                     : -1;
  rmm::device_uvector<float> codes(static_cast<std::size_t>(n), stream);
  thrust::transform(policy, labels, labels + n, found.codes.begin(),
                    codes.begin(), shifted<T>{unlabelled});
  return {std::move(codes), found.k - (has_unlabelled ? 1 : 0)};
}

template label_codes float_codes<int32_t>(const int32_t*, int64_t,
                                          rmm::cuda_stream_view);
template label_codes float_codes<int64_t>(const int64_t*, int64_t,
                                          rmm::cuda_stream_view);

}  // namespace cumlrkt::manifold
