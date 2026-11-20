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
#include "spectrum_enclosure.hpp"

namespace {

using cumlrkt::glm::condition_estimate;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::window_errors;
using cumlrkt::glm::window_spectrum;
using cumlrkt::glm::ritz::block_floor;
using cumlrkt::glm::ritz::certified_kappa;
using cumlrkt::glm::ritz::complement_floor;
using cumlrkt::glm::ritz::eigen_pairs;
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
constexpr int64_t window_columns = 4;
constexpr std::array<double, window_columns> window_values{0x1p-7, 1.0, 2.0,
                                                           3.0};
constexpr double window_mean = 1e-3;
constexpr double window_gram = 0.995;
constexpr double window_slack = 1e-12;
constexpr int64_t light_heavy_rows = int64_t{1} << 14;
constexpr double light_pair_eps = 0x1p-12;
constexpr float light_mean_weight = 0x1p-44F;
constexpr float light_plain_weight = 0x1p-20F;
constexpr double light_value = 5.0;
constexpr int64_t cluster_rows = int64_t{1} << 14;
constexpr double cluster_low_eps = 0x1p-10;
constexpr double cluster_next_eps = 0x1.6ap-6;

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

design two_pairs(int64_t n, double low_eps, double next_eps) {
  design c{std::vector<double>(n * pair_columns), n, pair_columns};
  for (int64_t i = 0; i < n; ++i) {
    c.x[i] = hadamard(i, 1);
    c.x[n + i] = hadamard(i, 1) + (low_eps * hadamard(i, 2));
    c.x[(2 * n) + i] = hadamard(i, 4);
    c.x[(3 * n) + i] = hadamard(i, 4) + (next_eps * hadamard(i, 8));
  }
  return c;
}

TEST(GramProof, TheExactPathAcceptsOnlyOnItsEnclosure) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const condition_estimate e = estimate_of(
      res, near_pair({tall_rows, pair_columns, near_eps}), refusal_kappa);
  EXPECT_LT(static_cast<double>(pair_kappa(near_eps)), refusal_kappa);
  EXPECT_EQ(e.path, gram_path::exact);
  EXPECT_GE(e.kappa, refusal_kappa);
  const condition_estimate cluster = estimate_of(
      res, two_pairs(cluster_rows, cluster_low_eps, cluster_next_eps),
      refusal_kappa);
  const auto kappa = static_cast<double>(pair_kappa(cluster_low_eps));
  EXPECT_EQ(cluster.path, gram_path::exact);
  EXPECT_EQ(cluster.window, 1);
  EXPECT_GE(cluster.kappa, kappa);
  EXPECT_LT(cluster.kappa, kappa * (1 + bound_slack));
  EXPECT_LT(cluster.kappa, refusal_kappa / 2);
}

struct window_case {
  std::vector<float> gram;
  eigen_pairs pairs;
  std::vector<double> basis;
  std::vector<double> applied;
  std::vector<double> crossed;
};

window_case diagonal_window() {
  const int64_t d = window_columns;
  window_case w{std::vector<float>(d * d, 0.0F),
                {{window_values.begin(), window_values.end()},
                 std::vector<double>(d * d, 0.0)},
                std::vector<double>(2 * d, 0.0),
                std::vector<double>(2 * d, 0.0),
                {window_values.front(), 0.0, 0.0, window_values.back()}};
  for (int64_t k = 0; k < d; ++k) {
    w.gram[(k * d) + k] = static_cast<float>(window_values.at(k));
    w.pairs.vectors[(k * d) + k] = 1.0;
  }
  w.basis[0] = 1.0;
  w.basis[(2 * d) - 1] = 1.0;
  w.applied[0] = window_values.front();
  w.applied[(2 * d) - 1] = window_values.back();
  return w;
}

spectrum_interval window_of(const resources& res, const window_case& w,
                            window_errors r) {
  const auto s =
      window_spectrum(raftrkt::handle_of(res.handle), w.gram, w.pairs,
                      {w.basis, w.applied, w.crossed, 1, 2}, r);
  EXPECT_TRUE(s.has_value());
  return s.value_or(spectrum_interval{0.0, 0.0});
}

TEST(GramProof, TheWindowBoundTakesEachErrorFromItsOwnTerm) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const window_case w = diagonal_window();
  const double h = window_values.front();
  const double beta = window_values.at(1);
  const auto mean =
      static_cast<double>(two_by_two(h - window_mean, beta, window_mean));
  const spectrum_interval m = window_of(res, w, {0.0, 0.0, window_mean});
  EXPECT_LE(m.low, mean);
  EXPECT_GE(m.low, mean - window_slack);
  const auto gram = static_cast<double>(two_by_two(h, beta - window_gram, 0));
  const spectrum_interval g = window_of(res, w, {window_gram, 0.0, 0.0});
  EXPECT_LE(g.low, gram);
  EXPECT_GE(g.low, gram - window_slack);
  EXPECT_GE(g.high, window_values.back() + window_gram);
}

struct light_column {
  design x;
  std::vector<float> w;
};

light_column light_pair(float weight) {
  const int64_t n = light_heavy_rows + 1;
  light_column c{{std::vector<double>(n * 3, 0.0), n, 3},
                 std::vector<float>(n, 1.0F)};
  for (int64_t i = 0; i < light_heavy_rows; ++i) {
    c.x.x[i] = hadamard(i, 1);
    c.x.x[n + i] = hadamard(i, 1) + (light_pair_eps * hadamard(i, 2));
    c.x.x[(2 * n) + i] = 1.0;
  }
  c.x.x[(3 * n) - 1] = light_value;
  c.w.back() = weight;
  return c;
}

condition_estimate weighted_estimate(const resources& res,
                                     const light_column& c) {
  const std::vector<float> host(c.x.x.begin(), c.x.x.end());
  const filled<float> x(res, host, {c.x.n * c.x.d});
  const filled<float> w(res, c.w, {c.x.n});
  return equilibrated_condition<float>(
      raftrkt::handle_of(res.handle),
      static_cast<const float*>(x.array.view.data),
      static_cast<const float*>(w.array.view.data), c.x.n, c.x.d, 0.0F, true,
      refusal_kappa);
}

TEST(GramProof, ALightColumnsMeanBoundPastLambdaMinStopsTheRitzProof) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const auto kappa = static_cast<double>(pair_kappa(light_pair_eps));
  const condition_estimate plain =
      weighted_estimate(res, light_pair(light_plain_weight));
  EXPECT_EQ(plain.path, gram_path::ritz);
  EXPECT_EQ(plain.window, 1);
  EXPECT_GE(plain.kappa, kappa);
  EXPECT_LT(plain.kappa, kappa * (1 + bound_slack));
  const condition_estimate e =
      weighted_estimate(res, light_pair(light_mean_weight));
  EXPECT_EQ(e.path, gram_path::exact);
  EXPECT_EQ(e.window, 1);
  EXPECT_GE(e.kappa, refusal_kappa);
}

}  // namespace
