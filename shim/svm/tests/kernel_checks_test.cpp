#include "kernel_checks.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/svm.h"

namespace cumlrkt::svm {
namespace {

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double rounds_to_most_float = 3.4028235e38;
constexpr double float_halfway = 0x1p128 - 0x1p103;

template <typename T>
bool accepted(const kernel_shape& k, double magnitude, double weight) {
  try {
    require_kernel<T>(k, magnitude, weight);
    return true;
  } catch (const logic_error&) {
    return false;
  }
}

double above(double v) {
  return std::nextafter(v, inf);
}

TEST(kernel_checks, the_limits_are_pinned) {
  EXPECT_EQ(most_kernel_exponent<float>, 0x1p20);
  EXPECT_EQ(most_kernel_exponent<double>, 0x1p49);
  EXPECT_EQ(most_kernel_value<float>, most_float / 4);
  EXPECT_EQ(most_kernel_value<double>, std::numeric_limits<double>::max() / 4);
  EXPECT_EQ(most_row_norm<float>, 0x1p123);
  EXPECT_EQ(most_row_norm<double>, 0x1p1019);
}

TEST(kernel_checks, rbf_and_sigmoid_bound_gamma_times_the_squared_norm) {
  for (const int32_t kernel : {CUML_SVM_KERNEL_RBF, CUML_SVM_KERNEL_SIGMOID}) {
    const kernel_shape k{kernel, 3, 0.5, 1.0};
    EXPECT_TRUE(accepted<float>(k, 0x1p21, 1e30));
    EXPECT_FALSE(accepted<float>(k, above(0x1p21), 1.0));
    EXPECT_TRUE(accepted<double>(k, 0x1p50, 1e300));
    EXPECT_FALSE(accepted<double>(k, above(0x1p50), 1.0));
  }
}

TEST(kernel_checks, linear_bounds_the_norm_times_the_weight) {
  const kernel_shape k{CUML_SVM_KERNEL_LINEAR, 3, 1.0, 0.0};
  const double most = most_float / 4;
  EXPECT_TRUE(accepted<float>(k, most / 0x1p10, 0x1p10));
  EXPECT_FALSE(accepted<float>(k, above(most / 0x1p10), 0x1p10));
  EXPECT_FALSE(accepted<float>(k, std::nan(""), 1.0));
}

TEST(kernel_checks, poly_bounds_the_power_times_the_weight) {
  const kernel_shape k{CUML_SVM_KERNEL_POLY, 2, 1.0, 1.0};
  EXPECT_EQ(kernel_bound<float>(k, 3.0, 5.0), 80.0);
  const kernel_shape steep{CUML_SVM_KERNEL_POLY, 40, 1.0, 1.0};
  EXPECT_FALSE(accepted<float>(steep, 20.0, 300.0));
  EXPECT_TRUE(accepted<double>(steep, 20.0, 300.0));
  const kernel_shape flat{CUML_SVM_KERNEL_POLY, 0, 1.0, 1e300};
  EXPECT_TRUE(accepted<double>(flat, 1e300, 1.0));
}

TEST(kernel_checks, the_refusal_names_its_remedy) {
  const kernel_shape k{CUML_SVM_KERNEL_RBF, 3, 1e30, 0.0};
  try {
    require_kernel<float>(k, 1.0, 1.0);
    FAIL() << "accepted";
  } catch (const logic_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "the kernel can overflow (bound 1e+30); lower #:gamma, #:degree "
              "or #:coef0, or rescale X");
  }
}

template <typename T>
bool narrowed(double v, bool positive) {
  try {
    require_narrowed<T>("C", v, positive);
    return true;
  } catch (const logic_error&) {
    return false;
  }
}

TEST(kernel_checks, parameters_are_checked_after_narrowing) {
  EXPECT_TRUE(narrowed<float>(most_float, true));
  EXPECT_TRUE(narrowed<float>(rounds_to_most_float, true));
  EXPECT_TRUE(narrowed<float>(std::nextafter(float_halfway, 0.0), true));
  EXPECT_FALSE(narrowed<float>(float_halfway, true));
  EXPECT_FALSE(narrowed<float>(1e39, true));
  EXPECT_FALSE(narrowed<float>(1e39, false));
  EXPECT_TRUE(narrowed<double>(1e39, true));
  EXPECT_FALSE(narrowed<float>(1e-50, true));
  EXPECT_TRUE(narrowed<float>(1e-50, false));
  EXPECT_TRUE(narrowed<double>(1e-50, true));
  EXPECT_FALSE(narrowed<double>(0.0, true));
  EXPECT_TRUE(narrowed<double>(0.0, false));
  EXPECT_FALSE(narrowed<double>(-1.0, true));
  EXPECT_TRUE(narrowed<double>(-1.0, false));
  EXPECT_FALSE(narrowed<double>(inf, false));
  EXPECT_FALSE(narrowed<double>(std::nan(""), false));
}

template <typename T>
bool cached(double cache_size, int64_t n) {
  try {
    require_cache<T>(cache_size, n);
    return true;
  } catch (const logic_error&) {
    return false;
  }
}

constexpr int64_t cache_rows = 300;
constexpr double overflowing_cache = 1e6;

template <typename T>
void expect_most_cache_size_is_the_bound(int64_t n) {
  EXPECT_TRUE(cached<T>(most_cache_size<T>(n), n)) << n;
  EXPECT_FALSE(cached<T>(2 * most_cache_size<T>(n), n)) << n;
}

TEST(kernel_checks, the_kernel_cache_index_fits_int32) {
  EXPECT_EQ(cache_elements<float>(1024.0, 300), int64_t{894784} * 300);
  EXPECT_TRUE(cached<float>(1024.0, 300));
  EXPECT_FALSE(cached<float>(16384.0, 300));
  EXPECT_FALSE(cached<double>(1e6, 300));
  EXPECT_FALSE(cached<double>(1e30, 300));
}

TEST(kernel_checks, the_cache_index_bound_is_not_monotone_in_rows) {
  EXPECT_LE(cache_elements<float>(8200.0, 100000), most_index);
  EXPECT_GT(cache_elements<float>(8200.0, 4000), most_index);
}

TEST(kernel_checks, most_cache_size_is_the_largest_cache_that_fits) {
  for (const int64_t n :
       {int64_t{1}, int64_t{300}, int64_t{569}, int64_t{20000}}) {
    expect_most_cache_size_is_the_bound<float>(n);
    expect_most_cache_size_is_the_bound<double>(n);
  }
}

TEST(kernel_checks, the_cache_refusal_names_its_remedy) {
  try {
    require_cache<double>(overflowing_cache, cache_rows);
    FAIL() << "accepted";
  } catch (const logic_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "cache-size: 1e+06 MiB overflows the kernel cache's index for "
              "300 rows; pass at most 16383");
  }
}

}  // namespace
}  // namespace cumlrkt::svm
