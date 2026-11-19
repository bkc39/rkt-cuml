#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "gram_condition.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"

namespace {

using cumlrkt::glm::condition_estimate;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::ritz::gram_bound;
using cumlrkt::glm::ritz::gram_error;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::gram_run_rows;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double refusal_kappa = 0x1p32;
constexpr int64_t sign_columns = 4;
constexpr int64_t sign_patterns = 6;
constexpr std::array<std::array<double, sign_columns>, sign_patterns> signs{
    {{1, 1, -1, -1},
     {1, -1, 1, -1},
     {1, -1, -1, 1},
     {-1, 1, 1, -1},
     {-1, 1, -1, 1},
     {-1, -1, 1, 1}}};
constexpr int64_t hadamard_rows = 1024;
constexpr double hadamard_loading = 0x1p16;
constexpr int hadamard_spread_bits = 10;
constexpr int64_t wide_columns = 100;

struct design {
  std::vector<double> x;
  int64_t n;
  int64_t d;
};

condition_estimate estimate_of(const resources& res, const design& c,
                               bool fit_intercept) {
  const std::vector<float> host(c.x.begin(), c.x.end());
  const filled<float> x(res, host, {c.n * c.d});
  return equilibrated_condition<float>(
      raftrkt::handle_of(res.handle),
      static_cast<const float*>(x.array.view.data), nullptr, c.n, c.d, 0.0F,
      fit_intercept, refusal_kappa);
}

void expect_refused(const resources& res, const design& c, bool fit_intercept) {
  const condition_estimate e = estimate_of(res, c, fit_intercept);
  EXPECT_NE(e.path, gram_path::certified) << c.n << " " << c.d;
  EXPECT_NE(e.path, gram_path::enclosed) << c.n << " " << c.d;
  EXPECT_GE(e.kappa, refusal_kappa) << c.n << " " << c.d;
}

design dummies_beside_ones(int64_t n) {
  design c{std::vector<double>(3 * n, 0.0), n, 3};
  for (int64_t i = 0; i < n; ++i) {
    c.x[((i % 2) * n) + i] = 1.0;
    c.x[(2 * n) + i] = 1.0;
  }
  return c;
}

design signs_summing_to_zero(int64_t n) {
  design c{std::vector<double>(sign_columns * n), n, sign_columns};
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t k = 0; k < sign_columns; ++k) {
      c.x[(k * n) + i] = signs.at(i % sign_patterns).at(k);
    }
  }
  return c;
}

double hadamard(int64_t i, int64_t j) {
  return std::popcount(static_cast<uint64_t>(i & j)) % 2 == 0 ? 1.0 : -1.0;
}

design hadamard_pair(int spread_bits) {
  const int64_t n = hadamard_rows;
  design c{std::vector<double>(2 * n), n, 2};
  for (int64_t j = 0; j < 2; ++j) {
    for (int64_t i = 0; i < n; ++i) {
      c.x[(j * n) + i] =
          (hadamard(i, j + 1) + (hadamard_loading * hadamard(i, n - 1))) *
          std::ldexp(1.0, spread_bits * static_cast<int>(j));
    }
  }
  return c;
}

TEST(GramError, DummiesBesideAOnesColumnAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {23379, 57115}) {
    expect_refused(res, dummies_beside_ones(n), false);
  }
}

TEST(GramError, SignColumnsSummingToZeroAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {4596, 4818}) {
    for (const bool fit_intercept : {false, true}) {
      expect_refused(res, signs_summing_to_zero(n), fit_intercept);
    }
  }
}

TEST(GramError, AHadamardPairPastTheThresholdIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int spread_bits : {0, hadamard_spread_bits}) {
    for (const bool fit_intercept : {false, true}) {
      expect_refused(res, hadamard_pair(spread_bits), fit_intercept);
    }
  }
}

TEST(GramError, TheBoundGrowsWithTheRunAndTheTrace) {
  const int64_t m = gram_run_rows(wide_columns);
  const gram_bound unit{
      .runs = 1, .run_rows = m, .trace = 1.0, .diagonal = 1.0};
  const double one = gram_error(unit);
  EXPECT_GT(one, static_cast<double>(m) * 0x1p-24);
  EXPECT_LT(one, static_cast<double>(m + 6) * 0x1p-24);
  EXPECT_GT(gram_error({.runs = 1,
                        .run_rows = m,
                        .trace = static_cast<double>(wide_columns),
                        .diagonal = 1.0}),
            static_cast<double>(wide_columns) * one * (1 - 0x1p-4));
  EXPECT_GT(
      gram_error({.runs = 1, .run_rows = 2 * m, .trace = 1.0, .diagonal = 1.0}),
      2 * static_cast<double>(m) * 0x1p-24);
}

TEST(GramError, NarrowAndWideKernelsNameTheirRuns) {
  EXPECT_EQ(gram_run_rows(1), cumlrkt::glm::ritz::narrow_run_rows);
  EXPECT_EQ(gram_run_rows(cumlrkt::glm::ritz::narrow_gram_columns),
            cumlrkt::glm::ritz::narrow_run_rows);
  EXPECT_LE(gram_run_rows(wide_columns), cumlrkt::glm::ritz::gram_block_rows);
}

}  // namespace
