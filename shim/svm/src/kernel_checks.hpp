#pragma once

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/svm.h"

namespace cumlrkt::svm {

template <typename T>
inline constexpr double unit_roundoff = std::numeric_limits<T>::epsilon() / 2;

template <typename T>
inline constexpr double most_kernel_value = most_value<T> / overflow_margin;

template <typename T>
inline constexpr double most_kernel_exponent = 1.0 / (16 * unit_roundoff<T>);

template <typename T>
inline constexpr double most_row_norm = squared_norm_bound<T> / overflow_margin;

struct kernel_shape {
  int32_t kernel;
  int32_t degree;
  double gamma;
  double coef0;
};

// magnitude: the largest |<x, x'>| the kernel can see (linear, poly), or the
// largest squared norm (rbf, sigmoid); weight: the total |coefficient| the
// kernel values are summed against.
template <typename T>
double kernel_bound(const kernel_shape& k, double magnitude, double weight) {
  switch (k.kernel) {
    case CUML_SVM_KERNEL_LINEAR:
      return magnitude * weight;
    case CUML_SVM_KERNEL_POLY:
      return std::pow((k.gamma * magnitude) + std::fabs(k.coef0), k.degree) *
             weight;
    default:
      return k.gamma * magnitude;
  }
}

template <typename T>
double kernel_limit(const kernel_shape& k) {
  return k.kernel == CUML_SVM_KERNEL_LINEAR || k.kernel == CUML_SVM_KERNEL_POLY
             ? most_kernel_value<T>
             : most_kernel_exponent<T>;
}

template <typename T>
void require_kernel(const kernel_shape& k, double magnitude, double weight) {
  const double bound = kernel_bound<T>(k, magnitude, weight);
  if (!(bound <= kernel_limit<T>(k))) {
    throw logic_error(std::format(
        "the kernel can overflow (bound {}); lower #:gamma, #:degree or "
        "#:coef0, or rescale X",
        printed(bound)));
  }
}

template <typename T>
void require_row_norms(const char* name, double largest, const T* x, int64_t n,
                       int64_t d, rmm::cuda_stream_view stream) {
  const double most = most_row_norm<T>;
  if (!(largest <= most)) {
    const auto [row, squared] = first_large_row(x, n, d, most, stream);
    throw logic_error(std::format(
        "{}: row {}: expected a squared norm of at most {}, got {}; rescale X",
        name, row, printed(most), printed(squared)));
  }
}

}  // namespace cumlrkt::svm
