#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "cd_staging.hpp"
#include "cumlrkt/support.hpp"
#include "residual_check.hpp"

namespace {

using cumlrkt::glm::accepts_zero;
using cumlrkt::glm::alpha_rounding;
using cumlrkt::glm::cd_epochs;
using cumlrkt::glm::column_sum_roundings;
using cumlrkt::glm::kkt_terms;
using cumlrkt::glm::moment;
using cumlrkt::glm::require_resolved;
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
  const double narrow_sigma = std::ldexp(1.0, (narrow_exponent - 2) / 2);
  EXPECT_DOUBLE_EQ(z.sigma, narrow_sigma);
  EXPECT_DOUBLE_EQ(z.sigma_y, std::ldexp(1.0, (spread_y_exponent - 2) / 2));
  EXPECT_DOUBLE_EQ(z.curvature, narrow_sigma * (1 + narrow_sigma));
  const zero_fit half =
      zero_fit_of<double>(wide_and_narrow(), {4, alpha, 0.5, rows});
  EXPECT_DOUBLE_EQ(half.curvature, (narrow_sigma * narrow_sigma) + (alpha / 2));
  EXPECT_FALSE(accepts_zero(z));
  const residual_terms wide{
      {moment{scaled_sum{1, 2}, scaled_sum{binding_cross, 2}}},
      {1},
      {1, spread_y_exponent}};
  EXPECT_TRUE(accepts_zero(zero_fit_of<double>(wide, {4, alpha, 1, rows})));
}

constexpr int binding_square_exponent = -18;
constexpr double binding_correlation = 0.9;
constexpr int binding_cross_exponent = -8;
constexpr double binding_sigma = 0x1p-10;
constexpr double small_ratio = 0.01;
constexpr double tiny_largest = 0x1p-16;
constexpr int stop_epochs = 4;
constexpr double stop_tol = 0.1;
constexpr double stop_residual = 0.15;
constexpr double far_factor = 10;

residual_terms binding_beside_orthogonal() {
  return {{moment{scaled_sum{1, binding_square_exponent},
                  scaled_sum{binding_correlation, binding_cross_exponent}},
           moment{scaled_sum{1, 2}, scaled_sum{0, 0}}},
          {1, 1},
          {1, 2}};
}

double expected_rounding(double l1_ratio, double alpha_max) {
  const double u = std::ldexp(1.0, -53);
  const double curvature =
      std::max((binding_sigma * binding_sigma) + (alpha_max * (1 - l1_ratio)),
               binding_sigma * (1 + binding_sigma));
  return u * std::sqrt(stop_epochs) *
         ((2 * tiny_largest * curvature) + (std::sqrt(rows) * binding_sigma)) /
         (alpha_max * l1_ratio);
}

const cd_epochs stop{stop_epochs, rows, stop_tol, std::ldexp(1.0, -53)};

TEST(CdZeroFit, TheAlphaRoundingIsTheBindingColumnsOwn) {
  for (const double ratio : {1.0, small_ratio}) {
    const double alpha_max = binding_correlation * binding_sigma / ratio;
    const zero_fit z = zero_fit_of<double>(binding_beside_orthogonal(),
                                           {4, alpha_max, ratio, rows});
    EXPECT_DOUBLE_EQ(z.alpha_max, alpha_max);
    EXPECT_DOUBLE_EQ(z.sigma, binding_sigma);
    EXPECT_DOUBLE_EQ(z.sigma_y, 1);
    EXPECT_DOUBLE_EQ(alpha_rounding(z, tiny_largest, ratio, stop),
                     expected_rounding(ratio, alpha_max));
  }
  EXPECT_EQ(alpha_rounding(zero_fit_of<double>(binding_beside_orthogonal(),
                                               {4, 1, 0, rows}),
                           tiny_largest, 0, stop),
            0);
}

template <typename F>
std::string refusal_of(F f) {
  try {
    f();
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(CdZeroFit, AnElasticNetAlphaFarOutsideTheBindingColumnsBandKeepsTheTol) {
  const double alpha_max = binding_correlation * binding_sigma / small_ratio;
  const zero_fit z = zero_fit_of<double>(binding_beside_orthogonal(),
                                         {4, alpha_max, small_ratio, rows});
  const double band = alpha_rounding(z, tiny_largest, small_ratio, stop);
  const kkt_terms unresolvable{stop_residual, stop_tol, 1};
  const auto advice = [&](double alpha) {
    return refusal_of([&] {
      require_resolved<double>(
          unresolvable, stop_epochs,
          {stop_tol, tiny_largest, alpha, z.alpha_max, band});
    });
  };
  const double true_band = expected_rounding(small_ratio, alpha_max);
  const std::string far = advice(alpha_max * (1 - (far_factor * true_band)));
  EXPECT_NE(far.find("lower tol below the largest coefficient or lower alpha"),
            std::string::npos)
      << far;
  const std::string near = advice(alpha_max * (1 - (true_band / 2)));
  EXPECT_NE(near.find("which no tol resolves"), std::string::npos) << near;
}

TEST(CdZeroFit, ZerosAreAcceptedUpToTheAllowance) {
  EXPECT_TRUE(accepts_zero({1e-14, 1e-14, 1, 1, 0, 0}));
  EXPECT_TRUE(accepts_zero({-1, 0, 1, 1, 0, 0}));
  EXPECT_FALSE(accepts_zero({std::nextafter(1e-14, 1.0), 1e-14, 1, 1, 0, 0}));
}

}  // namespace
