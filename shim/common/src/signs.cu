#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include "cumlrkt/device.cuh"
#include "cumlrkt/signs.hpp"

namespace cumlrkt {

namespace {

template <typename T>
struct entry {
  T value;
  int64_t row;
};

template <typename T>
struct column_entry {
  const T* x;
  int64_t n;
  int64_t k;

  __device__ entry<T> operator()(int64_t o) const {
    const int64_t c = o / n;
    const int64_t r = o - (c * n);
    return {x[(r * k) + c], r};
  }
};

struct column_of {
  int64_t n;

  __device__ int64_t operator()(int64_t o) const {
    return o / n;
  }
};

template <typename T>
struct first_largest {
  __device__ entry<T> operator()(const entry<T>& a, const entry<T>& b) const {
    const T ma = fabs(a.value);
    const T mb = fabs(b.value);
    return ma > mb || (ma == mb && a.row < b.row) ? a : b;
  }
};

template <typename T>
struct sign_ruled {
  T* x;
  const entry<T>* largest;
  int64_t k;

  __device__ void operator()(int64_t o) const {
    if (largest[o % k].value < 0) {
      x[o] = -x[o];
    }
  }
};

}  // namespace

template <typename T>
void columns_sign_rule(const raft::handle_t& handle, T* x, int64_t n,
                       int64_t k) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<entry<T>> largest(static_cast<std::size_t>(k), stream);
  const thrust::counting_iterator<int64_t> entries(0);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(entries, column_of{n}),
      thrust::make_transform_iterator(entries + (n * k), column_of{n}),
      thrust::make_transform_iterator(entries, column_entry<T>{x, n, k}),
      thrust::make_discard_iterator(), largest.data(),
      thrust::equal_to<int64_t>{}, first_largest<T>{});
  each(handle, n * k, sign_ruled<T>{x, largest.data(), k});
}

template void columns_sign_rule<float>(const raft::handle_t&, float*, int64_t,
                                       int64_t);

}  // namespace cumlrkt
