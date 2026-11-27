#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cumlrkt {

template <typename T>
struct dense_labels {
  rmm::device_uvector<int32_t> codes;
  rmm::device_uvector<T> values;
  int64_t k;
};

template <typename T>
dense_labels<T> dense_codes(const T* labels, int64_t n,
                            rmm::cuda_stream_view stream);

template <typename T>
void encode_against(const T* labels, int64_t n, const T* classes, int64_t k,
                    int32_t* codes, rmm::cuda_stream_view stream);

template <typename T>
int64_t first_non_integral(const T* labels, int64_t n,
                           rmm::cuda_stream_view stream);

}  // namespace cumlrkt
