#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/svm.h"
#include "kernel_checks.hpp"

namespace cumlrkt::svm {
namespace {

constexpr double inf = std::numeric_limits<double>::infinity();

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

}  // namespace
}  // namespace cumlrkt::svm
