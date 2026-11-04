#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt {

template <typename T>
void require_finite_elements(const char* name, const T* x, int64_t n,
                             rmm::cuda_stream_view stream);

template <typename T>
void require_non_negative_elements(const char* name, const T* x, int64_t n,
                                   rmm::cuda_stream_view stream);

template <typename T>
double require_sample_weights(const T* w, int64_t n,
                              rmm::cuda_stream_view stream);

}  // namespace cumlrkt
