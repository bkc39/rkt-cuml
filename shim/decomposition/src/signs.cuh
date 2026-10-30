#pragma once

#include <cstdint>

#include "cumlrkt/device.cuh"

namespace cumlrkt::decomposition {

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
