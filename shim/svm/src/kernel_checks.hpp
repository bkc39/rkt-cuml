#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// cuML narrows C, tol, gamma and coef0 to T, rounding to nearest; a double
// that passes as finite and positive can be +inf or 0 there.
template <typename T>
void require_narrowed(const char* key, double v, bool positive) {
  const T held = static_cast<T>(v);
  if (!std::isfinite(held) || (positive && !(held > T{0}))) {
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
  const int64_t whole_rows = (most_index / n) * n;
  return std::floor(static_cast<double>(whole_rows) *
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

inline void require_positive_cache(double cache_size) {
  if (!(cache_size > 0)) {
    throw logic_error(std::format(
        "cache-size: expected a positive value, got {}", printed(cache_size)));
  }
}

struct predict_shape {
  int64_t rows;
  int64_t support;
};

// svcPredictX (cuml svm/svc_impl.cuh) mirrored in T: every row in one tile when
// rows·support·sizeof(T) fits buffer_size·2^20, else
// ⌊buffer/(support·sizeof(T))⌋ rows, at least one. cuVS's KernelOp views the
// n_batch × support tile through an int index.
template <typename T>
int64_t predict_tile_elements(double cache_size, predict_shape shape) {
  const T buffer = static_cast<T>(cache_size) * 1024 * 1024;
  const std::size_t row_bytes =
      static_cast<std::size_t>(shape.support) * sizeof(T);
  int64_t n_batch = shape.rows;
  if (static_cast<T>(static_cast<std::size_t>(shape.rows) * row_bytes) >
      buffer) {
    n_batch = std::max<int64_t>(
        1, static_cast<int64_t>(buffer / static_cast<T>(row_bytes)));
  }
  return n_batch * shape.support;
}

template <typename T>
void require_predict_tile(double cache_size, predict_shape shape) {
  if (predict_tile_elements<T>(cache_size, shape) > most_index) {
    throw logic_error(std::format(
        "X: the kernel tile for {} rows against {} support vectors overflows "
        "an int index; predict on at most {} rows at a time",
        shape.rows, shape.support, most_index / shape.support));
  }
}

}  // namespace cumlrkt::svm
