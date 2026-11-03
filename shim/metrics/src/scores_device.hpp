#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>

namespace cumlrkt::metrics {

struct sample_weights {
  const float* single;
  const double* wide;
  int exponent;
};

struct matched {
  double correct;
  double total;
};

template <typename T, typename P>
double r2_of(const T* y, const P* p, int64_t n, int exponent, sample_weights w,
             rmm::cuda_stream_view stream);

template <typename T, typename P>
double squared_error_of(const T* y, const P* p, int64_t n, int64_t d,
                        int exponent, sample_weights w,
                        rmm::cuda_stream_view stream);

template <typename A, typename B>
matched matches_of(const A* a, const B* b, int64_t n, sample_weights w,
                   rmm::cuda_stream_view stream);

}  // namespace cumlrkt::metrics
