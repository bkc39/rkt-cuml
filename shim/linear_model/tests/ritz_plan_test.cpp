#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"
#include "spectrum_enclosure.hpp"
#include "symmetric_eigen.hpp"

namespace {

using cumlrkt::glm::eigen_of;
using cumlrkt::glm::enclosed_spectrum;
using cumlrkt::glm::enclosure_norms_of;
using cumlrkt::glm::ritz::certified_kappa;
using cumlrkt::glm::ritz::dot;
using cumlrkt::glm::ritz::eigen_pairs;
using cumlrkt::glm::ritz::enclosure;
using cumlrkt::glm::ritz::enclosure_pays;
using cumlrkt::glm::ritz::exact_gram_cheaper;
using cumlrkt::glm::ritz::gram_error;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::gram_shape;
using cumlrkt::glm::ritz::plan_of;
using cumlrkt::glm::ritz::spectrum_error;
using cumlrkt::glm::ritz::spectrum_interval;
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
constexpr int64_t correlated_rows = 20000;
constexpr int64_t correlated_columns = 100;
constexpr double correlated_loading = 33.0;
constexpr gram_shape correlated_shape{1000000, correlated_columns};
constexpr uint64_t correlated_salts = 4;
constexpr double skew_cosine = 0.6;

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
  EXPECT_EQ(plan_of(values, plan_shape, 1.5 * plan.kappa).path,
            gram_path::ritz);
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

struct factor_shape {
  int64_t n;
  int64_t d;
  double loading;
  uint64_t salt = factor_salt;
};

std::vector<double> factor_draws(factor_shape f) {
  const int64_t n = f.n;
  const int64_t d = f.d;
  std::vector<double> x(n * (d + 1));
  uint64_t s = (f.salt * lcg_increment) + 1;
  for (double& v : x) {
    s = (s * lcg_multiplier) + lcg_increment;
    v = static_cast<double>((s >> lcg_shift) % draw_modulus) - draw_centre;
  }
  for (int64_t k = 1; k <= d; ++k) {
    for (int64_t i = 0; i < n; ++i) {
      x[(k * n) + i] += f.loading * x[i];
    }
  }
  x.erase(x.begin(), x.begin() + n);
  return x;
}

std::vector<double> factor_columns_of(double loading) {
  const int64_t n = factor_rows;
  std::vector<double> x = factor_draws({n, factor_columns, loading});
  for (int64_t i = 0; i < n; ++i) {
    x[(2 * n) + i] = x[i] + x[n + i] - x[(3 * n) + i];
  }
  return x;
}

std::vector<float> equilibrated_gram(std::vector<double> x, int64_t n,
                                     int64_t d) {
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

struct residual_measure {
  double residual;
  double orthogonality;
};

double spectral_norm(const resources& res, const std::vector<double>& m,
                     int64_t d) {
  const std::optional<eigen_pairs> e =
      eigen_of(raftrkt::handle_of(res.handle), m, d, false);
  return e ? std::max(std::abs(e->values.front()), std::abs(e->values.back()))
           : std::numeric_limits<double>::infinity();
}

residual_measure measured(const resources& res, const std::vector<float>& g,
                          const eigen_pairs& e, int64_t d) {
  double worst = 0.0;
  std::vector<double> w(d * d);
  for (int64_t j = 0; j < d; ++j) {
    const double* v = e.vectors.data() + (j * d);
    double r = 0.0;
    for (int64_t i = 0; i < d; ++i) {
      double gv = 0.0;
      for (int64_t k = 0; k < d; ++k) {
        gv += static_cast<double>(g[(k * d) + i]) * v[k];
      }
      r += (gv - (e.values[j] * v[i])) * (gv - (e.values[j] * v[i]));
      w[(j * d) + i] =
          dot(e.vectors.data() + (i * d), v, d) - (i == j ? 1.0 : 0.0);
    }
    worst = std::max(worst, std::sqrt(r));
  }
  return {worst, spectral_norm(res, w, d)};
}

TEST(GramPlan, TheSpectrumErrorCoversSsyevdsResiduals) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  for (const double loading : {3.0, 10.0}) {
    const std::vector<float> g = equilibrated_gram(factor_columns_of(loading),
                                                   factor_rows, factor_columns);
    const std::optional<eigen_pairs> e =
        eigen_of(handle, g, factor_columns, true);
    if (!e.has_value()) {
      ADD_FAILURE() << loading;
      continue;
    }
    const eigen_pairs& pairs = *e;
    const residual_measure m = measured(res, g, pairs, factor_columns);
    EXPECT_LT(m.residual + (std::abs(pairs.values.back()) * m.orthogonality),
              spectrum_error(pairs.values, factor_columns) / 2)
        << loading;
    EXPECT_GE(
        enclosure_norms_of(handle, g, pairs, factor_columns).orthogonality,
        m.orthogonality)
        << loading;
  }
}

std::vector<double> promoted(const std::vector<float>& g, int64_t d) {
  std::vector<double> m(g.begin(), g.end());
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = j + 1; i < d; ++i) {
      m[(i * d) + j] = m[(j * d) + i];
    }
  }
  return m;
}

void expect_certified_by_residuals(const resources& res, uint64_t salt) {
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  const int64_t d = correlated_columns;
  const std::vector<float> g = equilibrated_gram(
      factor_draws({correlated_rows, d, correlated_loading, salt}),
      correlated_rows, d);
  const std::optional<eigen_pairs> e = eigen_of(handle, g, d, true);
  const std::optional<eigen_pairs> exact =
      eigen_of(handle, promoted(g, d), d, false);
  if (!e || !exact) {
    ADD_FAILURE() << salt;
    return;
  }
  const auto plan = plan_of(e->values, correlated_shape, 0x1p32);
  EXPECT_NE(plan.path, gram_path::certified) << salt;
  EXPECT_TRUE(enclosure_pays(e->values, plan, correlated_shape, 0x1p32))
      << salt;
  const std::optional<spectrum_interval> s =
      enclosed_spectrum(handle, g, *e, d);
  if (!s) {
    ADD_FAILURE() << salt;
    return;
  }
  EXPECT_LT(s->low, exact->values.front()) << salt;
  EXPECT_GT(s->high, exact->values.back()) << salt;
  EXPECT_TRUE(
      certified_kappa({s->low - gram_error(d), s->high + gram_error(d)}, 0x1p32)
          .has_value())
      << salt;
}

TEST(GramPlan, ACorrelatedSpectrumIsCertifiedByItsResiduals) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < correlated_salts; ++salt) {
    expect_certified_by_residuals(res, salt);
  }
}

TEST(GramPlan, ASkewBasisCannotHideAnEigenvalue) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const double sine = std::sqrt(1.0 - (skew_cosine * skew_cosine));
  const std::vector<float> g{1.0F, 0.0F, 0.0F, 3.0F};
  const eigen_pairs e{{1.0, 1.0}, {1.0, 0.0, skew_cosine, sine}};
  const std::optional<spectrum_interval> s =
      enclosed_spectrum(raftrkt::handle_of(res.handle), g, e, 2);
  ASSERT_TRUE(s.has_value());
  EXPECT_LE(s->low, 1.0);
  EXPECT_GE(s->high, 3.0);
}

TEST(GramPlan, ZeroComputedResidualsStillCoverTheirOwnRounding) {
  const double matrix = std::sqrt(5.0);
  const double basis = std::sqrt(2.0);
  const std::optional<spectrum_interval> s = enclosure({1.0, 2.0},
                                                       {.symmetric = 0.0,
                                                        .orthogonality = 0.0,
                                                        .matrix = matrix,
                                                        .basis = basis,
                                                        .product = matrix},
                                                       2);
  const double rounding = 2 * 0x1p-53 * basis * ((basis * matrix) + matrix);
  ASSERT_TRUE(s.has_value());
  EXPECT_LE(s->low, 1.0 - rounding);
  EXPECT_GE(s->high, 2.0 + rounding);
}

}  // namespace
