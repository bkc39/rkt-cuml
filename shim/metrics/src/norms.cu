#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "norms.hpp"

namespace cumlrkt::metrics {

namespace {

template <typename T>
struct normless {
  const T* x;
  int64_t d;

  __device__ bool operator()(int64_t i) const {
    const T* row = x + i * d;
    T sum{0};
    for (int64_t j = 0; j < d; ++j) {
      sum += row[j] * row[j];
    }
    return sum == T{0};
  }
};

}  // namespace

template <typename T>
int64_t first_normless_row(const T* x, int64_t n, int64_t d,
                           rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const int64_t i = *thrust::find_if(rmm::exec_policy(stream), rows, rows + n,
                                     normless<T>{x, d});
  return i == n ? -1 : i;
}

template int64_t first_normless_row<float>(const float*, int64_t, int64_t,
                                           rmm::cuda_stream_view);
template int64_t first_normless_row<double>(const double*, int64_t, int64_t,
                                            rmm::cuda_stream_view);

}  // namespace cumlrkt::metrics
