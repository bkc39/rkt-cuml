#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <vector>

#include "gram_condition.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"

namespace {

using cumlrkt::glm::condition_estimate;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::ritz::block_floor;
using cumlrkt::glm::ritz::certified_kappa;
using cumlrkt::glm::ritz::complement_floor;
using cumlrkt::glm::ritz::enclosure_terms;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::spectrum_interval;
using cumlrkt::glm::ritz::widened;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double refusal_kappa = 0x1p32;
constexpr int64_t pair_columns = 4;
constexpr int64_t window_rows = int64_t{1} << 14;
constexpr int64_t tall_rows = int64_t{1} << 22;
constexpr double window_eps = 0x1p-10;
constexpr double tall_eps = 0x1p-14;
constexpr double near_eps = 0x1.4p-15;
constexpr double bound_slack = 0x1p-10;
constexpr double tall_slack = 0x1p-6;
constexpr double floor_slack = 0x1p-40;
constexpr double floor_eta = 0x1p-10;
constexpr double floor_error = 0x1p-20;
constexpr int64_t light_rows = 4;
constexpr double light_kappa = 2.0;
constexpr std::array<std::array<double, 3>, 4> blocks{
    {{1e-6, 1.0, 1e-4}, {1.0, 1e-6, 1e-4}, {0.5, 0.5, 0.25}, {1e-3, 2.0, 0.0}}};
constexpr std::array<std::array<double, 2>, 3> rounded_ends{
    {{1.0, 0x1p-60}, {0x1.8p0, 0x1.8p-58}, {0x1.fffffffffffffp-1, 0x1p-62}}};
constexpr double third_low = 3.0;

struct design {
  std::vector<double> x;
  int64_t n;
  int64_t d;
};

double hadamard(int64_t i, int64_t j) {
  return std::popcount(static_cast<uint64_t>(i & j)) % 2 == 0 ? 1.0 : -1.0;
}

struct pair_shape {
  int64_t n;
  int64_t d;
  double eps;
};

design near_pair(pair_shape p) {
  const auto [n, d, eps] = p;
  design c{std::vector<double>(n * d), n, d};
  for (int64_t i = 0; i < n; ++i) {
    c.x[i] = hadamard(i, 1);
    c.x[n + i] = hadamard(i, 1) + (eps * hadamard(i, 2));
    for (int64_t k = 2; k < d; ++k) {
      c.x[(k * n) + i] = hadamard(i, int64_t{1} << (k + 1));
    }
  }
  return c;
}

long double pair_kappa(double eps) {
  const long double one = 1;
  const long double rho =
      one / std::sqrt(one + (static_cast<long double>(eps) * eps));
  return (one + rho) / (one - rho);
}

condition_estimate estimate_of(const resources& res, const design& c,
                               double limit) {
  const std::vector<float> host(c.x.begin(), c.x.end());
  const filled<float> x(res, host, {c.n * c.d});
  return equilibrated_condition<float>(
      raftrkt::handle_of(res.handle),
      static_cast<const float*>(x.array.view.data), nullptr, c.n, c.d, 0.0F,
      true, limit);
}

long double two_by_two(long double h, long double b, long double c) {
  const long double apart = (b - h) / 2;
  return ((h + b) / 2) - std::sqrt((apart * apart) + (c * c));
}

TEST(GramProof, TheBlockFloorIsTheTwoByTwosSmallestEigenvalue) {
  for (const auto& [h, b, c] : blocks) {
    const auto exact = static_cast<double>(two_by_two(h, b, c));
    const double floor = block_floor({h, b, c});
    EXPECT_LE(floor, exact) << h << " " << b << " " << c;
    EXPECT_GE(floor, exact - (floor_slack * std::abs(exact))) << h << " " << b;
  }
}

TEST(GramProof, EveryIntervalEndRoundsOutward) {
  for (const auto& [end, error] : rounded_ends) {
    const spectrum_interval s = widened({end, end}, error);
    const long double low = static_cast<long double>(end) - error;
    const long double high = static_cast<long double>(end) + error;
    EXPECT_LT(static_cast<long double>(end + error), high) << end;
    EXPECT_GT(static_cast<long double>(end - error), low) << end;
    EXPECT_LE(static_cast<long double>(s.low), low) << end << " " << error;
    EXPECT_GE(static_cast<long double>(s.high), high) << end << " " << error;
  }
  const auto k = certified_kappa({third_low, 1.0}, 1.0);
  ASSERT_TRUE(k.has_value());
  EXPECT_GE(static_cast<long double>(*k), 1.0L / 3.0L);
}

TEST(GramProof, TheComplementFloorLeansOnTheBasissSkew) {
  const std::vector<double> values{1e-6, 1.0, 2.0};
  const enclosure_terms t{.error = floor_error, .eta = floor_eta};
  const double lean = (values[1] - values[0]) * floor_eta * floor_eta;
  const double expected = (values[1] - lean - floor_error) / (1 + floor_eta);
  const double floor = complement_floor(values, 1, t);
  EXPECT_LE(floor, expected);
  EXPECT_GE(floor, expected * (1 - floor_slack));
}

TEST(GramProof, AWindowWhoseBlockBoundHoldsIsAcceptedOnRitz) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const condition_estimate e = estimate_of(
      res, near_pair({window_rows, pair_columns, window_eps}), refusal_kappa);
  const auto kappa = static_cast<double>(pair_kappa(window_eps));
  EXPECT_EQ(e.path, gram_path::ritz);
  EXPECT_EQ(e.window, 1);
  EXPECT_GE(e.kappa, kappa);
  EXPECT_LT(e.kappa, kappa * (1 + bound_slack));
}

TEST(GramProof, AWindowTheBlockBoundCannotProveGoesExact) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const condition_estimate e = estimate_of(
      res, near_pair({tall_rows, pair_columns, tall_eps}), refusal_kappa);
  const auto kappa = static_cast<double>(pair_kappa(tall_eps));
  EXPECT_EQ(e.path, gram_path::exact);
  EXPECT_GE(e.kappa, kappa);
  EXPECT_LT(e.kappa, kappa * (1 + tall_slack));
}

TEST(GramProof, AColumnSetByOneLightRowIsProvenThroughItsResidual) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<float> host{0, 1, 2, 3, 1, 1, 1, 5};
  const std::vector<float> weights{3, 3, 3, 0x1p-148F};
  const filled<float> x(res, host, {light_rows * 2});
  const filled<float> w(res, weights, {light_rows});
  const condition_estimate e = equilibrated_condition<float>(
      raftrkt::handle_of(res.handle),
      static_cast<const float*>(x.array.view.data),
      static_cast<const float*>(w.array.view.data), light_rows, 2, 0.0F, true,
      refusal_kappa);
  EXPECT_LT(e.kappa, light_kappa);
}

TEST(GramProof, TheExactPathAcceptsOnlyOnItsEnclosure) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const condition_estimate e = estimate_of(
      res, near_pair({tall_rows, pair_columns, near_eps}), refusal_kappa);
  EXPECT_LT(static_cast<double>(pair_kappa(near_eps)), refusal_kappa);
  EXPECT_EQ(e.path, gram_path::exact);
  EXPECT_GE(e.kappa, refusal_kappa);
}

}  // namespace
