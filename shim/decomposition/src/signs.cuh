#pragma once

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

namespace cumlrkt::decomposition {

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

template <typename T>
struct signed_components {
  const T* components;
  T* out;
  int64_t m;
  int64_t d;
  bool* flipped;

  __device__ void operator()(int64_t i) const {
    T best = components[i];
    for (int64_t j = 1; j < d; ++j) {
      const T c = components[i + (j * m)];
      if (fabs(c) > fabs(best)) {
        best = c;
      }
    }
    const bool flip = best < 0;
    for (int64_t j = 0; j < d; ++j) {
      const T c = components[i + (j * m)];
      out[(i * d) + j] = flip ? -c : c;
    }
    if (flipped != nullptr) {
      flipped[i] = flip;
    }
  }
};

}  // namespace cumlrkt::decomposition
