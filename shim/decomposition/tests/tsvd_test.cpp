#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cumlrkt/decomposition.h"
#include "decomposition_fixture.hpp"
#include "raftrkt/c_api.h"
#include "tsvd_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 8;
constexpr int64_t cols = 3;
constexpr int32_t fit_regions = 4;
constexpr int32_t transformed_regions = 5;
constexpr double first_variance = 4;
constexpr double second_variance = 1;
constexpr double third_variance = 0.25;
constexpr double first_square = 32;
constexpr double second_square = 8;
constexpr double third_square = 2;

template <typename T>
std::vector<T> axis_rows() {
  return {4, 0, 0, -4, 0, 0,  0, 2, 0, 0, -2, 0,
          0, 0, 1, 0,  0, -1, 0, 0, 0, 0, 0,  0};
}

template <typename T>
std::vector<T> identity() {
  return {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

template <typename T>
void expect_axis_spectrum(const record& out) {
  const std::vector<T> variance{static_cast<T>(first_variance),
                                static_cast<T>(second_variance),
                                static_cast<T>(third_variance)};
  const T scale =
      static_cast<T>(1) /
      static_cast<T>(first_variance + second_variance + third_variance);
  std::vector<T> ratio;
  for (const T v : variance) {
    ratio.push_back(v * scale);
  }
  EXPECT_EQ(out.read<T>(1),
            (std::vector<T>{std::sqrt(static_cast<T>(first_square)),
                            std::sqrt(static_cast<T>(second_square)),
                            std::sqrt(static_cast<T>(third_square))}));
  EXPECT_EQ(out.read<T>(2), variance);
  EXPECT_EQ(out.read<T>(3), ratio);
}

template <typename T>
void expect_axis_fit(int32_t solver) {
  const resources res;
  const std::vector<T> host = axis_rows<T>();
  const filled<T> x(res, host, {rows, cols});
  record out(res);
  ASSERT_EQ(tsvd_fit_into(
                res, x.view(), out,
                {.n_components = cols, .solver = solver, .with_transform = 1}),
            "accepted");
  ASSERT_EQ(out.r.count, transformed_regions);
  EXPECT_EQ(out.read<T>(0), identity<T>());
  expect_axis_spectrum<T>(out);
  EXPECT_EQ(out.read<T>(4), host);
  record plain(res);
  ASSERT_EQ(tsvd_fit_into(res, x.view(), plain, {.n_components = 2}),
            "accepted");
  ASSERT_EQ(plain.r.count, fit_regions);
  EXPECT_EQ(plain.read<T>(0), (std::vector<T>{1, 0, 0, 0, 1, 0}));
  record t(res);
  ASSERT_EQ(tsvd_apply_into(res, cuml_decomposition_tsvd_transform, x.view(),
                            &out.r.regions[0], t),
            "accepted");
  EXPECT_EQ(t.read<T>(0), host);
  record back(res);
  ASSERT_EQ(tsvd_apply_into(res, cuml_decomposition_tsvd_inverse_transform,
                            &t.r.regions[0], &out.r.regions[0], back),
            "accepted");
  EXPECT_EQ(back.read<T>(0), host);
}

TEST(Tsvd, AxisFixtureFitsAndTransformsExactly) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t solver :
       {CUML_DECOMPOSITION_SOLVER_FULL, CUML_DECOMPOSITION_SOLVER_JACOBI}) {
    expect_axis_fit<float>(solver);
    expect_axis_fit<double>(solver);
  }
}

template <typename T>
std::vector<T> duplicate_columns(int64_t n) {
  std::vector<T> host;
  for (int64_t i = 0; i < n; ++i) {
    const auto a = static_cast<T>(std::sin(static_cast<double>(i) + 0.5));
    const auto b = static_cast<T>(std::cos(static_cast<double>(3 * i)));
    const auto c = static_cast<T>(std::sin(static_cast<double>(7 * i) + 1));
    host.insert(host.end(), {a, b, a, c, a, b});
  }
  return host;
}

template <typename T>
void expect_finite_spectrum(const record& out) {
  for (int32_t r = 1; r < fit_regions; ++r) {
    for (const T v : out.read<T>(r)) {
      EXPECT_TRUE(std::isfinite(v)) << "region " << r;
      EXPECT_GE(v, 0) << "region " << r;
    }
  }
}

template <typename T>
void expect_rank_deficient(int32_t solver) {
  constexpr int64_t n = 30;
  constexpr int64_t d = 6;
  const resources res;
  const filled<T> x(res, duplicate_columns<T>(n), {n, d});
  record out(res);
  ASSERT_EQ(
      tsvd_fit_into(res, x.view(), out, {.n_components = d, .solver = solver}),
      "accepted");
  expect_finite_spectrum<T>(out);
  for (const T v : out.read<T>(0)) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(Tsvd, RankDeficientSingularValuesAreZeroNotNan) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t solver :
       {CUML_DECOMPOSITION_SOLVER_FULL, CUML_DECOMPOSITION_SOLVER_JACOBI}) {
    expect_rank_deficient<float>(solver);
    expect_rank_deficient<double>(solver);
  }
}

template <typename T>
void expect_positive_zeros(const std::vector<T>& values) {
  for (const T v : values) {
    EXPECT_EQ(v, 0);
    EXPECT_FALSE(std::signbit(v));
  }
}

template <typename T>
void expect_constant(T value) {
  constexpr int64_t n = 6;
  constexpr int64_t d = 4;
  const resources res;
  const filled<T> x(res, std::vector<T>(n * d, value), {n, d});
  record out(res);
  ASSERT_EQ(tsvd_fit_into(res, x.view(), out, {.n_components = 2}), "accepted");
  expect_positive_zeros(out.read<T>(2));
  expect_positive_zeros(out.read<T>(3));
  if (value == 0) {
    expect_positive_zeros(out.read<T>(1));
  }
  for (const T v : out.read<T>(0)) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(Tsvd, ConstantRowsExplainNoVariance) {
  CUMLRKT_REQUIRE_GPU();
  expect_constant<float>(0);
  expect_constant<double>(0);
  expect_constant<float>(2);
  expect_constant<double>(2);
}

TEST(Tsvd, ViewsOnAnotherDeviceAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, axis_rows<double>(), {rows, cols});
  record fitted(res);
  ASSERT_EQ(tsvd_fit_into(res, x.view(), fitted, {.n_components = 2}),
            "accepted");
  rr_view x_elsewhere = *x.view();
  x_elsewhere.device = 1;
  rr_view components = fitted.r.regions[0];
  components.device = 1;
  record out(res);
  EXPECT_EQ(tsvd_fit_into(res, &x_elsewhere, out),
            "X: on device 1, but the resources are on device 0");
  for (const tsvd_apply entry : {cuml_decomposition_tsvd_transform,
                                 cuml_decomposition_tsvd_inverse_transform}) {
    EXPECT_EQ(
        tsvd_apply_into(res, entry, &x_elsewhere, &fitted.r.regions[0], out),
        "X: on device 1, but the resources are on device 0");
    EXPECT_EQ(tsvd_apply_into(res, entry, x.view(), &components, out),
              "components: on device 1, but the resources are on device 0");
  }
  expect_untouched(out);
}

TEST(Tsvd, OutputsAreFiniteJustUnderTheOverflowBound) {
  CUMLRKT_REQUIRE_GPU();
  constexpr int64_t n = 50;
  constexpr int64_t d = 4;
  constexpr std::size_t period = 7;
  constexpr float under_bound = 0x1p62F;
  const resources res;
  std::vector<float> host(static_cast<std::size_t>(n * d));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(i % period) - static_cast<float>(i % 3);
  }
  host[(3 * d) + 1] = under_bound;
  const filled<float> x(res, host, {n, d});
  for (const int32_t solver :
       {CUML_DECOMPOSITION_SOLVER_FULL, CUML_DECOMPOSITION_SOLVER_JACOBI}) {
    record out(res);
    ASSERT_EQ(tsvd_fit_into(
                  res, x.view(), out,
                  {.n_components = 2, .solver = solver, .with_transform = 1}),
              "accepted");
    for (int32_t i = 0; i < transformed_regions; ++i) {
      for (const float v : out.read<float>(i)) {
        EXPECT_TRUE(std::isfinite(v)) << "region " << i;
      }
    }
  }
}

}  // namespace
}  // namespace cumlrkt::test
