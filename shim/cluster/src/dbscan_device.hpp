#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::cluster {

template <typename T>
int64_t first_zero_row(const T* x, int64_t n, int64_t d,
                       rmm::cuda_stream_view stream);

template <typename T>
void unit_rows(T* x, int64_t n, int64_t d, rmm::cuda_stream_view stream);

template <typename Index>
int64_t core_count(const Index* core, int64_t n, rmm::cuda_stream_view stream);

}  // namespace cumlrkt::cluster
