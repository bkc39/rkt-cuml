#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include "linkage_device.hpp"

namespace cumlrkt::cluster {

namespace {

template <typename Index>
struct ascending_pair {
  Index* children;

  __device__ void operator()(int64_t i) const {
    Index* pair = children + 2 * i;
    if (pair[0] > pair[1]) {
      const Index first = pair[0];
      pair[0] = pair[1];
      pair[1] = first;
    }
  }
};

}  // namespace

template <typename Index>
void ascending_pairs(Index* children, int64_t rows,
                     rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> first(0);
  thrust::for_each(rmm::exec_policy(stream), first, first + rows,
                   ascending_pair<Index>{children});
}

template void ascending_pairs<int32_t>(int32_t*, int64_t,
                                       rmm::cuda_stream_view);
template void ascending_pairs<int64_t>(int64_t*, int64_t,
                                       rmm::cuda_stream_view);

}  // namespace cumlrkt::cluster
