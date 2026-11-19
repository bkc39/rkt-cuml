#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"
#include "symmetric_eigen.hpp"

namespace {

using cumlrkt::glm::eigen_of;
using cumlrkt::glm::ritz::eigen_pairs;
using cumlrkt::glm::ritz::exact_gram_cheaper;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::gram_shape;
using cumlrkt::glm::ritz::plan_of;
using cumlrkt::glm::ritz::spectrum_error;
using cumlrkt::test::resources;

constexpr double near_step = 0x1p-30;
constexpr int64_t plan_columns = 200;
constexpr gram_shape plan_shape{1000000, plan_columns};
constexpr int64_t plan_near = 40;
constexpr int64_t plan_wide = 120;
constexpr int64_t factor_rows = 2000;
constexpr int64_t factor_columns = 500;
constexpr uint64_t factor_salt = 7;
constexpr uint64_t lcg_multiplier = 6364136223846793005ULL;
constexpr uint64_t lcg_increment = 1442695040888963407ULL;
constexpr int lcg_shift = 33;
constexpr uint64_t draw_modulus = 23;
constexpr double draw_centre = 11.0;

struct spectrum_of {
  int64_t near;
  double low = 0.0;
  double next = 0.5;
};

std::vector<double> spectrum(spectrum_of f) {
  std::vector<double> values(plan_columns, 1.0);
  values[0] = f.low;
  for (int64_t j = 1; j < f.near; ++j) {
    values[j] = f.low + (near_step * static_cast<double>(j));
  }
  values[f.near] = f.next;
  return values;
}

TEST(GramPlan, AWellSeparatedSpectrumIsCertifiedWithoutRefining) {
  const std::vector<double> values = spectrum({.near = 1, .low = 0x1p-4});
  const auto plan = plan_of(values, plan_shape, 0x1p32);
  EXPECT_EQ(plan.path, gram_path::certified);
  EXPECT_GT(plan.kappa, 1.0 / 0x1p-4);
  EXPECT_LT(plan.kappa, (1.0 / 0x1p-4) * (1 + 0x1p-8));
  EXPECT_EQ(plan_of(values, plan_shape, 0.0).path, gram_path::ritz);
  EXPECT_EQ(plan_of(values, plan_shape, plan.kappa).path, gram_path::ritz);
}

TEST(GramPlan, TheWindowHoldsEveryEigenvalueWithinReach) {
  const std::vector<double> values = spectrum({.near = plan_near});
  const auto plan = plan_of(values, plan_shape, 0x1p32);
  EXPECT_EQ(plan.path, gram_path::ritz);
  EXPECT_EQ(plan.window, plan_near);
  EXPECT_LT(near_step * plan_near, plan.reach);
}

TEST(GramPlan, NoGapAtTheWindowsEdgeTakesTheExactGram) {
  const auto edge =
      plan_of(spectrum({.near = plan_near}), plan_shape, 0x1p32).reach;
  for (const double next : {1.5 * edge, 1.99 * edge}) {
    const auto plan = plan_of(spectrum({.near = plan_near, .next = next}),
                              plan_shape, 0x1p32);
    EXPECT_EQ(plan.path, gram_path::exact) << next / edge;
    EXPECT_EQ(plan.window, plan_near) << next / edge;
  }
  const auto below =
      plan_of(spectrum({.near = plan_near, .low = -edge}), plan_shape, 0x1p32);
  EXPECT_EQ(below.path, gram_path::exact);
  EXPECT_EQ(plan_of(spectrum({.near = plan_near, .next = 2.01 * edge}),
                    plan_shape, 0x1p32)
                .path,
            gram_path::ritz);
}

TEST(GramPlan, AWindowThatCostsMoreThanTheExactGramTakesIt) {
  const auto wide = plan_of(spectrum({.near = plan_wide}), plan_shape, 0x1p32);
  EXPECT_EQ(wide.path, gram_path::exact);
  EXPECT_EQ(wide.window, plan_wide);
  EXPECT_TRUE(exact_gram_cheaper(plan_wide + 1, plan_shape));
  EXPECT_FALSE(exact_gram_cheaper(plan_near + 1, plan_shape));
}

std::vector<double> factor_columns_of(double loading) {
  const int64_t n = factor_rows;
  const int64_t d = factor_columns;
  std::vector<double> x(n * (d + 1));
  uint64_t s = (factor_salt * lcg_increment) + 1;
  for (double& v : x) {
    s = (s * lcg_multiplier) + lcg_increment;
    v = static_cast<double>((s >> lcg_shift) % draw_modulus) - draw_centre;
  }
  for (int64_t k = 1; k <= d; ++k) {
    for (int64_t i = 0; i < n; ++i) {
      x[(k * n) + i] += loading * x[i];
    }
  }
  x.erase(x.begin(), x.begin() + n);
  for (int64_t i = 0; i < n; ++i) {
    x[(2 * n) + i] = x[i] + x[n + i] - x[(3 * n) + i];
  }
  return x;
}

std::vector<float> equilibrated_gram(std::vector<double> x) {
  const int64_t n = factor_rows;
  const int64_t d = factor_columns;
  for (int64_t k = 0; k < d; ++k) {
    double* c = x.data() + (k * n);
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      mean += c[i];
    }
    mean /= static_cast<double>(n);
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      c[i] -= mean;
      ss += c[i] * c[i];
    }
    const double scale = 1.0 / std::sqrt(ss);
    for (int64_t i = 0; i < n; ++i) {
      c[i] *= scale;
    }
  }
  std::vector<float> g(d * d);
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t k = 0; k <= j; ++k) {
      double s = 0.0;
      for (int64_t i = 0; i < n; ++i) {
        s += x[(j * n) + i] * x[(k * n) + i];
      }
      g[(j * d) + k] = static_cast<float>(s);
      g[(k * d) + j] = static_cast<float>(s);
    }
  }
  return g;
}

double largest_residual(const std::vector<float>& g, const eigen_pairs& e) {
  const int64_t d = factor_columns;
  double worst = 0.0;
  const double top = std::abs(e.values.back());
  for (int64_t j = 0; j < d; ++j) {
    const double* v = e.vectors.data() + (j * d);
    double r = 0.0;
    double norm = 0.0;
    for (int64_t i = 0; i < d; ++i) {
      double gv = 0.0;
      for (int64_t k = 0; k < d; ++k) {
        gv += static_cast<double>(g[(k * d) + i]) * v[k];
      }
      r += (gv - (e.values[j] * v[i])) * (gv - (e.values[j] * v[i]));
      norm += v[i] * v[i];
    }
    worst = std::max(worst, std::sqrt(r) + (top * std::abs(norm - 1.0)));
  }
  return worst;
}

TEST(GramPlan, TheSpectrumErrorCoversSsyevdsResiduals) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const double loading : {3.0, 10.0}) {
    const std::vector<float> g = equilibrated_gram(factor_columns_of(loading));
    const std::optional<eigen_pairs> e =
        eigen_of(raftrkt::handle_of(res.handle), g, factor_columns, true);
    if (!e.has_value()) {
      ADD_FAILURE() << loading;
      continue;
    }
    const eigen_pairs& pairs = *e;
    EXPECT_LT(largest_residual(g, pairs),
              spectrum_error(pairs.values, factor_columns) / 2)
        << loading;
  }
}

}  // namespace
