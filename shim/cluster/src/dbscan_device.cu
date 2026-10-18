#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "dbscan_device.hpp"

namespace cumlrkt::cluster {

namespace {

template <typename T>
struct zero_row {
  const T* x;
  int64_t d;

  __device__ bool operator()(int64_t i) const {
    const T* row = x + i * d;
    for (int64_t j = 0; j < d; ++j) {
      if (!(row[j] * row[j] == T{0})) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace

template <typename T>
int64_t first_zero_row(const T* x, int64_t n, int64_t d,
                       rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const int64_t i = *thrust::find_if(rmm::exec_policy(stream), rows, rows + n,
                                     zero_row<T>{x, d});
  return i == n ? -1 : i;
}

template <typename Index>
int64_t core_count(const Index* core, int64_t n, rmm::cuda_stream_view stream) {
  return thrust::find(rmm::exec_policy(stream), core, core + n, Index{-1}) -
         core;
}

template int64_t first_zero_row<float>(const float*, int64_t, int64_t,
                                       rmm::cuda_stream_view);
template int64_t first_zero_row<double>(const double*, int64_t, int64_t,
                                        rmm::cuda_stream_view);
template int64_t core_count<int32_t>(const int32_t*, int64_t,
                                     rmm::cuda_stream_view);
template int64_t core_count<int64_t>(const int64_t*, int64_t,
                                     rmm::cuda_stream_view);

}  // namespace cumlrkt::cluster
