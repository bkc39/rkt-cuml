#pragma once

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <type_traits>

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

template <typename T>
inline constexpr const char* float_name =
    std::is_same_v<T, float> ? "float32" : "float64";

// cuML narrows C, tol, gamma and coef0 to T; a double that passes as finite
// and positive can be +inf or 0 there.
template <typename T>
void require_narrowed(const char* key, double v, bool positive) {
  const bool finite = std::fabs(v) <= most_value<T>;
  if (!finite || (positive && !(static_cast<T>(v) > T{0}))) {
    throw logic_error(std::format("{}: expected a {} finite in {}, got {}", key,
                                  positive ? "positive value" : "value",
                                  float_name<T>, printed(v)));
  }
}

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

inline constexpr int64_t cache_associativity = 32;

inline constexpr double mebibyte = 1024.0 * 1024.0;

// raft::cache::Cache (raft/util/cache.cuh) keeps int(MiB·2^20 / (sizeof(T)·n))
// vectors, rounded down to whole sets, and sizes its store as vectors·n in
// int: past most_index that wraps, and a wrapped positive size is a store too
// small for the writes into it.
template <typename T>
int64_t cache_elements(double cache_size, int64_t n) {
  const float vectors = (static_cast<float>(cache_size) * 1024 * 1024) /
                        static_cast<float>(sizeof(T) * n);
  if (!(vectors < static_cast<float>(most_index))) {
    return std::numeric_limits<int64_t>::max();
  }
  const auto whole = static_cast<int64_t>(vectors);
  return (whole / cache_associativity) * cache_associativity * n;
}

template <typename T>
double most_cache_size(int64_t n) {
  return std::floor(static_cast<double>((most_index / n) * n) *
                    static_cast<double>(sizeof(T)) / mebibyte);
}

template <typename T>
void require_cache(double cache_size, int64_t n) {
  if (cache_elements<T>(cache_size, n) > most_index) {
    throw logic_error(std::format(
        "cache-size: {} MiB overflows the kernel cache's index for {} rows; "
        "pass at most {}",
        printed(cache_size), n, printed(most_cache_size<T>(n))));
  }
}

}  // namespace cumlrkt::svm
