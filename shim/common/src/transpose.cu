#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/device.cuh"
#include "cumlrkt/transpose.hpp"

namespace cumlrkt {

namespace {

template <typename T>
struct column_major {
  const T* x;
  const T* variance;
  T* out;
  int64_t n;
  int64_t d;
  int exponent;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / n;
    const int64_t i = o - (j * n);
    const T v = scaled(x[(i * d) + j], exponent);
    out[o] = variance == nullptr ? v : times(v, root(variance[j]));
  }
};

template <typename T>
struct row_major {
  const T* x;
  const T* variance;
  T* out;
  int64_t n;
  int64_t d;

  __device__ void operator()(int64_t o) const {
    const int64_t i = o / d;
    const int64_t j = o - (i * d);
    const T v = x[i + (j * n)];
    out[o] = variance == nullptr ? v : over(v, divisor(variance[j]));
  }
};

}  // namespace

template <typename T>
void to_columns(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const T* variance, T* out, int exponent) {
  each(handle, n * d, column_major<T>{x, variance, out, n, d, exponent});
}

template <typename T>
void to_rows(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
             const T* variance, T* out) {
  each(handle, n * d, row_major<T>{x, variance, out, n, d});
}

template void to_columns<float>(const raft::handle_t&, const float*, int64_t,
                                int64_t, const float*, float*, int);
template void to_columns<double>(const raft::handle_t&, const double*, int64_t,
                                 int64_t, const double*, double*, int);
template void to_rows<float>(const raft::handle_t&, const float*, int64_t,
                             int64_t, const float*, float*);
template void to_rows<double>(const raft::handle_t&, const double*, int64_t,
                              int64_t, const double*, double*);

}  // namespace cumlrkt
