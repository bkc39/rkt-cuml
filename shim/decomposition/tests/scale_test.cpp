#include "cumlrkt/scale.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cumlrkt::test {
namespace {

using cumlrkt::scale_exponent;

TEST(ScaleExponent,
     TheLeastUnscaledMagnitudeIsTheSquareRootOfTheLeastNormalOverEpsilon) {
  EXPECT_EQ(std::ldexp(1.0, least_unscaled_exponent<float>),
            std::sqrt(static_cast<double>(std::numeric_limits<float>::min())) /
                static_cast<double>(std::numeric_limits<float>::epsilon()));
  EXPECT_EQ(std::ldexp(1.0, least_unscaled_exponent<double>),
            std::sqrt(std::numeric_limits<double>::min()) /
                std::numeric_limits<double>::epsilon());
}

TEST(ScaleExponent, DataInTheNormalRangeIsNotScaled) {
  for (const double largest : {0x1p-40, 1.0, 0x1p100, 3e38}) {
    EXPECT_EQ(scale_exponent<float>(largest), 0) << largest;
  }
  for (const double largest : {0x1p-459, 1e-100, 1.0, 1e300}) {
    EXPECT_EQ(scale_exponent<double>(largest), 0) << largest;
  }
}

TEST(ScaleExponent, NoMagnitudeAtAllIsNotScaled) {
  EXPECT_EQ(scale_exponent<float>(0.0), 0);
  EXPECT_EQ(scale_exponent<double>(0.0), 0);
}

constexpr double denormal_float = std::numeric_limits<float>::denorm_min();
constexpr double denormal_double = std::numeric_limits<double>::denorm_min();
constexpr std::array<double, 5> small_floats{0x1.fffffep-41, 1e-20, 1e-25,
                                             1e-40, denormal_float};
constexpr std::array<double, 5> small_doubles{0x1.fffffffffffffp-460, 1e-160,
                                              1e-165, 1e-320, denormal_double};

template <typename T, std::size_t N>
void expect_one_to_two(const std::array<double, N>& magnitudes) {
  for (const double largest : magnitudes) {
    const double scaled = std::ldexp(largest, scale_exponent<T>(largest));
    EXPECT_GE(scaled, 1.0) << largest;
    EXPECT_LT(scaled, 2.0) << largest;
  }
}

TEST(ScaleExponent, SmallDataLandsInOneToTwo) {
  expect_one_to_two<float>(small_floats);
  expect_one_to_two<double>(small_doubles);
  EXPECT_EQ(scale_exponent<float>(denormal_float), 149);
  EXPECT_EQ(scale_exponent<double>(denormal_double), 1074);
}

}  // namespace
}  // namespace cumlrkt::test
