#pragma once

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cfloat>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

namespace cumlrkt {

template <typename F>
void each(const raft::handle_t& handle, int64_t count, F f) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), count, f);
}

inline __device__ float scaled(float v, int exponent) {
  return scalbnf(v, exponent);
}
inline __device__ double scaled(double v, int exponent) {
  return scalbn(v, exponent);
}
inline __device__ float root(float v) {
  return __fsqrt_rn(v);
}
inline __device__ double root(double v) {
  return __dsqrt_rn(v);
}
inline __device__ float times(float a, float b) {
  return __fmul_rn(a, b);
}
inline __device__ double times(double a, double b) {
  return __dmul_rn(a, b);
}
inline __device__ float over(float a, float b) {
  return __fdiv_rn(a, b);
}
inline __device__ double over(double a, double b) {
  return __ddiv_rn(a, b);
}
inline __device__ float epsilon(float) {
  return FLT_EPSILON;
}
inline __device__ double epsilon(double) {
  return DBL_EPSILON;
}

template <typename T>
__device__ T divisor(T variance) {
  const T scale = root(variance);
  return scale < epsilon(T{}) ? epsilon(T{}) : scale;
}

}  // namespace cumlrkt
