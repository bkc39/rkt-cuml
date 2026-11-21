#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "cd_staging.hpp"
#include "residual_check.hpp"

namespace {

using cumlrkt::glm::accepts_zero;
using cumlrkt::glm::column_sum_roundings;
using cumlrkt::glm::moment;
using cumlrkt::glm::residual_terms;
using cumlrkt::glm::scaled_sum;
using cumlrkt::glm::zero_fit;
using cumlrkt::glm::zero_fit_of;

constexpr int64_t rows = 100;

residual_terms one_column(double cross) {
  return {{moment{scaled_sum{1, 2}, scaled_sum{cross, 2}}}, {1}, {1, 2}};
}

TEST(CdZeroFit, TheAllowanceIsTheResidualsRoundingBound) {
  const zero_fit z = zero_fit_of<double>(one_column(0.75), {4, 0.5, 0.5, rows});
  const double count =
      (5.0 * static_cast<double>(column_sum_roundings(rows))) + 14;
  const double u = std::ldexp(1.0, -53);
  EXPECT_EQ(column_sum_roundings(rows), 268);
  EXPECT_DOUBLE_EQ(z.residual, 0.75 - 0.25);
  EXPECT_DOUBLE_EQ(z.allowance,
                   (count * u / (1 - (count * u))) + (2 * u * 0.25));
  EXPECT_DOUBLE_EQ(z.alpha_max, 1.5);
  EXPECT_EQ(column_sum_roundings(int64_t{2048} * 256), 268);
  EXPECT_EQ(column_sum_roundings((int64_t{2048} * 256) + 1), 269);
  const zero_fit narrow =
      zero_fit_of<float>(one_column(0.75), {4, 1.5, 0.5, rows});
  EXPECT_DOUBLE_EQ(narrow.allowance - z.allowance,
                   2 * (std::ldexp(1.0, -24) * 0.75 - u * 0.25));
  EXPECT_TRUE(std::isinf(
      zero_fit_of<double>(one_column(0.75), {4, 0.5, 0, rows}).alpha_max));
}

constexpr int narrow_exponent = -38;
constexpr double binding_cross = 0.875;
constexpr int spread_y_exponent = 42;

residual_terms wide_and_narrow() {
  return {
      {moment{scaled_sum{1, 2}, scaled_sum{0, 0}},
       moment{scaled_sum{1, narrow_exponent}, scaled_sum{binding_cross, 2}}},
      {1, 1},
      {1, spread_y_exponent}};
}

TEST(CdZeroFit, EachColumnIsHeldToItsOwnSpread) {
  const double gap = std::ldexp(1.0, -30);
  const double alpha = binding_cross - gap;
  const zero_fit z =
      zero_fit_of<double>(wide_and_narrow(), {4, alpha, 1, rows});
  const double count =
      (5.0 * static_cast<double>(column_sum_roundings(rows))) + 14;
  const double u = std::ldexp(1.0, -53);
  EXPECT_DOUBLE_EQ(z.residual, gap);
  EXPECT_DOUBLE_EQ(z.allowance,
                   (count * u / (1 - (count * u))) + (2 * u * alpha));
  EXPECT_DOUBLE_EQ(z.alpha_max, binding_cross);
  EXPECT_FALSE(accepts_zero(z));
  const residual_terms wide{
      {moment{scaled_sum{1, 2}, scaled_sum{binding_cross, 2}}},
      {1},
      {1, spread_y_exponent}};
  EXPECT_TRUE(accepts_zero(zero_fit_of<double>(wide, {4, alpha, 1, rows})));
}

TEST(CdZeroFit, ZerosAreAcceptedUpToTheAllowance) {
  EXPECT_TRUE(accepts_zero({1e-14, 1e-14, 1, 1}));
  EXPECT_TRUE(accepts_zero({-1, 0, 1, 1}));
  EXPECT_FALSE(accepts_zero({std::nextafter(1e-14, 1.0), 1e-14, 1, 1}));
}

}  // namespace
