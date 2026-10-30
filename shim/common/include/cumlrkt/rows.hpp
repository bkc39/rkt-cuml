#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

namespace cumlrkt {

template <typename T>
std::pair<int64_t, double> first_large_row(const T* x, int64_t n, int64_t d,
                                           double most,
                                           rmm::cuda_stream_view stream);

template <typename T>
std::pair<int64_t, T> first_normless_row(const T* x, int64_t n, int64_t d,
                                         T least, rmm::cuda_stream_view stream);

template <typename T>
int64_t first_zero_row(const T* x, int64_t n, int64_t d,
                       rmm::cuda_stream_view stream);

template <typename T>
int64_t first_non_finite(const T* x, int64_t count,
                         rmm::cuda_stream_view stream);

template <typename T>
int64_t first_large_element(const T* x, int64_t count, double most,
                            rmm::cuda_stream_view stream);

template <typename T>
double largest_magnitude(const T* x, int64_t count,
                         rmm::cuda_stream_view stream);

template <typename T>
int64_t first_row_unlike_first(const T* x, int64_t n, int64_t d,
                               rmm::cuda_stream_view stream);

}  // namespace cumlrkt
