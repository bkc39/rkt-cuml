#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::svm {

template <typename T>
T element_variance(const T* x, int64_t count, rmm::cuda_stream_view stream);

template <typename T>
double scale_gamma(const T* x, int64_t n, int64_t d,
                   rmm::cuda_stream_view stream);

}  // namespace cumlrkt::svm
