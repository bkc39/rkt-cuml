#include "gram_condition.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"

namespace {

using cumlrkt::glm::checks_conditioning;
using cumlrkt::glm::condition_estimate;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::glm_family;
using cumlrkt::glm::glm_solver;
using cumlrkt::glm::kappa_limit;
using cumlrkt::glm::require_conditioned;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::pass_chunk;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

struct columns {
  std::vector<double> x;
  int64_t n;
  int64_t d;
};

struct request {
  double limit = 0.0;
  int64_t chunk = pass_chunk;
  bool fit_intercept = true;
};

template <typename T>
condition_estimate estimate_of(const resources& res, const columns& c,
                               const std::vector<double>& w, double alpha,
                               request r) {
  const std::vector<T> host(c.x.begin(), c.x.end());
  const filled<T> x(res, host, {c.n * c.d});
  const std::vector<T> wh(w.begin(), w.end());
  const filled<T> weights(res, wh.empty() ? std::vector<T>{T{1}} : wh,
                          {wh.empty() ? 1 : c.n});
  return equilibrated_condition<T>(
      raftrkt::handle_of(res.handle), static_cast<const T*>(x.array.view.data),
      w.empty() ? nullptr : static_cast<const T*>(weights.array.view.data), c.n,
      c.d, static_cast<T>(alpha), r.fit_intercept, r.limit, r.chunk);
}

template <typename T>
double condition_of(const resources& res, const columns& c,
                    const std::vector<double>& w, double alpha,
                    bool fit_intercept) {
  return estimate_of<T>(res, c, w, alpha, {.fit_intercept = fit_intercept})
      .kappa;
}

constexpr int64_t draw_step = 37;
constexpr int64_t draw_offset = 11;
constexpr int64_t draw_modulus = 23;
constexpr double draw_centre = 11.0;
constexpr int64_t scramble_divisor = 5;
constexpr int64_t pair_count = 5;
constexpr int64_t second_salt = 20;
constexpr int64_t third_salt = 40;
constexpr int64_t fourth_salt = 60;
constexpr int64_t salts = 8;
constexpr int64_t weighted_rows = 61;
constexpr int64_t last_row = weighted_rows - 1;
constexpr double outlier = 5.0;

double draw(int64_t i, int64_t salt) {
  return static_cast<double>(
             ((i * (draw_step + (2 * salt))) + (draw_offset * salt)) %
             draw_modulus) -
         draw_centre;
}

columns summed(int64_t n, double eps) {
  columns c{std::vector<double>(n * 3), n, 3};
  for (int64_t i = 0; i < n; ++i) {
    c.x[i] = draw(i, 0);
    c.x[n + i] = draw(i, 1);
    c.x[(2 * n) + i] = c.x[i] + c.x[n + i] + (eps * draw(i, 2));
  }
  return c;
}

double scrambled(int64_t i, int64_t salt) {
  return static_cast<double>(((i * (draw_step + (2 * salt))) +
                              (draw_offset * salt) +
                              ((i * i) / (scramble_divisor + salt))) %
                             draw_modulus) -
         draw_centre;
}

columns summed_beside_a_near_pair(int64_t n, int64_t salt, double eps) {
  columns c{std::vector<double>(n * pair_count), n, pair_count};
  for (int64_t i = 0; i < n; ++i) {
    c.x[i] = scrambled(i, salt);
    c.x[n + i] = scrambled(i, salt + second_salt);
    c.x[(2 * n) + i] = c.x[i] + c.x[n + i];
    c.x[(3 * n) + i] = scrambled(i, salt + third_salt);
    c.x[(4 * n) + i] =
        c.x[(3 * n) + i] + (eps * scrambled(i, salt + fourth_salt));
  }
  return c;
}

constexpr uint64_t lcg_multiplier = 6364136223846793005ULL;
constexpr uint64_t lcg_increment = 1442695040888963407ULL;
constexpr int lcg_shift = 33;
constexpr int64_t many_rows = 5000;
constexpr int64_t ritz_rows = 20000;
constexpr int64_t many_pairs = 30;
constexpr int64_t spare_columns = 120;
constexpr int64_t many_salts = 8;
constexpr double many_eps = 0x1p-12;
constexpr int64_t cluster_pairs = 6;
constexpr double cluster_eps = 0x1.4p-5;
constexpr int64_t correlated_rows = 20000;
constexpr int64_t correlated_columns = 100;
constexpr double correlated_loading = 14.0;
constexpr uint64_t correlated_salts = 4;
constexpr int64_t tall_rows = int64_t{1} << 20;

struct draw_stream {
  int64_t n;
  uint64_t salt;
};

struct near_pairs {
  int64_t pairs;
  int64_t spare;
};

std::vector<double> integers(draw_stream d) {
  std::vector<double> v(d.n);
  uint64_t s = (d.salt * lcg_increment) + 1;
  for (double& x : v) {
    s = (s * lcg_multiplier) + lcg_increment;
    x = static_cast<double>((s >> lcg_shift) % draw_modulus) - draw_centre;
  }
  return v;
}

columns near_pairs_after(std::vector<std::vector<double>> cols, draw_stream d,
                         near_pairs shape, double eps) {
  const int64_t n = d.n;
  for (int64_t j = 0; j < shape.pairs + shape.spare; ++j) {
    const uint64_t at = d.salt + (2 * (static_cast<uint64_t>(j) + 1));
    std::vector<double> x = integers({n, at});
    cols.push_back(x);
    if (j < shape.pairs) {
      const std::vector<double> z = integers({n, at + 1});
      for (int64_t i = 0; i < n; ++i) {
        x[i] += eps * z[i];
      }
      cols.push_back(x);
    }
  }
  columns c{{}, n, static_cast<int64_t>(cols.size())};
  for (const std::vector<double>& col : cols) {
    c.x.insert(c.x.end(), col.begin(), col.end());
  }
  return c;
}

columns summed_beside_near_pairs(draw_stream d, near_pairs shape,
                                 double eps = many_eps) {
  std::vector<std::vector<double>> cols{integers({d.n, d.salt}),
                                        integers({d.n, d.salt + 1})};
  std::vector<double> sum(d.n);
  for (int64_t i = 0; i < d.n; ++i) {
    sum[i] = cols[0][i] + cols[1][i];
  }
  cols.push_back(sum);
  return near_pairs_after(std::move(cols), d, shape, eps);
}

columns beside_near_pairs(draw_stream d, near_pairs shape) {
  return near_pairs_after({}, d, shape, many_eps);
}

columns twins(int64_t n) {
  columns c{std::vector<double>(n * 2), n, 2};
  for (int64_t i = 0; i < n; ++i) {
    c.x[i] = draw(i, 3);
    c.x[n + i] = c.x[i];
  }
  return c;
}

double spread_of(const columns& c) {
  double mean = 0.0;
  for (int64_t i = 0; i < c.n; ++i) {
    mean += c.x[i];
  }
  mean /= static_cast<double>(c.n);
  double s = 0.0;
  for (int64_t i = 0; i < c.n; ++i) {
    s += (c.x[i] - mean) * (c.x[i] - mean);
  }
  return s;
}

std::string refusal_of(double kappa, glm_solver solver, bool single) {
  try {
    if (single) {
      require_conditioned<float>(kappa, solver);
    } else {
      require_conditioned<double>(kappa, solver);
    }
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(GramCondition, OrthogonalCentredColumnsHaveConditionOne) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const columns c{{3, 1, 3, 1, 5, 5, 1, 1}, 4, 2};
  EXPECT_NEAR(condition_of<float>(res, c, {}, 0.0, true), 1.0, 1e-6);
  EXPECT_NEAR(condition_of<double>(res, c, {}, 0.0, true), 1.0, 1e-12);
}

TEST(GramCondition, WithoutAnInterceptTheColumnsAreNotCentred) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const columns c{{1, 1, 1, 1, 1, 2, 3, 4}, 4, 2};
  const double r = 10.0 / std::sqrt(120.0);
  EXPECT_NEAR(condition_of<double>(res, c, {}, 0.0, false), (1 + r) / (1 - r),
              1e-9);
}

TEST(GramCondition, AlphaBoundsTheConditionOfTwinColumns) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {int64_t{60}, int64_t{70001}}) {
    const columns c = twins(n);
    const double s = spread_of(c);
    const double alpha = s / 8;
    const double kappa = ((2 * s) + alpha) / alpha;
    EXPECT_NEAR(condition_of<float>(res, c, {}, alpha, true), kappa,
                kappa * 1e-6)
        << n;
    EXPECT_NEAR(condition_of<double>(res, c, {}, alpha, true), kappa,
                kappa * 1e-6)
        << n;
  }
}

TEST(GramCondition, ExactlyCollinearFloat32ColumnsArePastTheThreshold) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {int64_t{60}, int64_t{70001}}) {
    const double kappa = condition_of<float>(res, summed(n, 0), {}, 0.0, true);
    EXPECT_GT(kappa, 0x1p40) << n;
    EXPECT_LT(condition_of<float>(res, summed(n, 0x1p-7), {}, 0.0, true),
              0x1p24)
        << n;
  }
}

TEST(GramCondition, ExactCollinearityBesideANearPairIsPastTheThreshold) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {int64_t{1000}, int64_t{5000}}) {
    for (const double eps : {0x1p-12, 0x1p-13}) {
      for (int64_t salt = 0; salt < salts; ++salt) {
        EXPECT_GT(
            condition_of<float>(res, summed_beside_a_near_pair(n, salt, eps),
                                {}, 0.0, true),
            0x1p32)
            << n << " " << eps << " " << salt;
      }
    }
  }
}

void expect_refused_through_the_window(const resources& res, int64_t spare,
                                       uint64_t salt) {
  const columns c =
      summed_beside_near_pairs({ritz_rows, salt * 1000}, {many_pairs, spare});
  const condition_estimate e =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
  EXPECT_EQ(e.path, gram_path::ritz) << c.d << " " << salt;
  EXPECT_EQ(e.window, many_pairs + 1) << c.d << " " << salt;
  EXPECT_GT(e.kappa, 0x1p32) << c.d << " " << salt;
}

TEST(GramCondition, ManyNearPairsBesideExactCollinearityArePastTheThreshold) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t spare : {int64_t{0}, spare_columns}) {
    for (uint64_t salt = 0; salt < many_salts; ++salt) {
      expect_refused_through_the_window(res, spare, salt);
    }
  }
}

void expect_chunks_match_one_pass(const resources& res, uint64_t salt) {
  const columns c =
      beside_near_pairs({ritz_rows, salt * 1000}, {many_pairs, spare_columns});
  const condition_estimate chunked =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
  const condition_estimate single =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32, .chunk = c.d + 1});
  EXPECT_EQ(chunked.path, gram_path::ritz) << salt;
  EXPECT_GT(chunked.window, pass_chunk) << salt;
  EXPECT_NEAR(chunked.kappa, single.kappa, single.kappa * 0x1p-30) << salt;
}

TEST(GramCondition, AWindowWiderThanOneChunkMatchesASinglePass) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < many_salts; ++salt) {
    expect_chunks_match_one_pass(res, salt);
  }
}

void expect_certified_above_the_refined_kappa(const resources& res,
                                              uint64_t salt) {
  const columns c =
      beside_near_pairs({many_rows, salt * 1000}, {0, spare_columns});
  const condition_estimate certified =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
  const condition_estimate refined = estimate_of<float>(res, c, {}, 0.0, {});
  EXPECT_EQ(certified.path, gram_path::certified) << salt;
  EXPECT_NE(refined.path, gram_path::certified) << salt;
  EXPECT_GE(certified.kappa, refined.kappa) << salt;
  EXPECT_LT(certified.kappa, refined.kappa * (1 + 0x1p-6)) << salt;
}

TEST(GramCondition, ConditionedColumnsAreCertifiedFromTheFloat32Spectrum) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < many_salts; ++salt) {
    expect_certified_above_the_refined_kappa(res, salt);
  }
}

TEST(GramCondition, CollinearityBesideAnUnseparatedClusterTakesTheExactGram) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < many_salts; ++salt) {
    const columns c = summed_beside_near_pairs({many_rows, salt * 1000},
                                               {cluster_pairs, 0}, cluster_eps);
    const condition_estimate e =
        estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
    EXPECT_EQ(e.path, gram_path::exact) << salt;
    EXPECT_LT(e.window, c.d - 1) << salt;
    EXPECT_GT(e.kappa, 0x1p32) << salt;
  }
}

columns correlated(draw_stream d, int64_t width) {
  const std::vector<double> f = integers(d);
  columns c{{}, d.n, width};
  for (int64_t j = 0; j < width; ++j) {
    std::vector<double> x =
        integers({d.n, d.salt + 1 + static_cast<uint64_t>(j)});
    for (int64_t i = 0; i < d.n; ++i) {
      x[i] += correlated_loading * f[i];
    }
    c.x.insert(c.x.end(), x.begin(), x.end());
  }
  return c;
}

void expect_enclosed_above_the_refined_kappa(const resources& res,
                                             uint64_t salt) {
  const columns c =
      correlated({correlated_rows, salt * 1000}, correlated_columns);
  const condition_estimate enclosed =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
  const condition_estimate refined = estimate_of<float>(res, c, {}, 0.0, {});
  EXPECT_TRUE(enclosed.path == gram_path::certified ||
              enclosed.path == gram_path::enclosed)
      << salt;
  EXPECT_EQ(enclosed.window, 0) << salt;
  EXPECT_EQ(refined.path, gram_path::exact) << salt;
  EXPECT_GE(enclosed.kappa, refined.kappa) << salt;
  EXPECT_LT(enclosed.kappa, 4 * refined.kappa) << salt;
}

TEST(GramCondition, CorrelatedColumnsAreCertifiedByTheirResiduals) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < correlated_salts; ++salt) {
    expect_enclosed_above_the_refined_kappa(res, salt);
  }
}

void expect_tall_gram_certified_near_the_exact(const resources& res,
                                               uint64_t salt) {
  const columns c = beside_near_pairs({tall_rows, salt * 1000}, {0, 2});
  const condition_estimate certified =
      estimate_of<float>(res, c, {}, 0.0, {.limit = 0x1p32});
  const condition_estimate exact = estimate_of<float>(res, c, {}, 0.0, {});
  EXPECT_EQ(certified.path, gram_path::certified) << salt;
  EXPECT_EQ(exact.path, gram_path::exact) << salt;
  EXPECT_GE(certified.kappa, exact.kappa) << salt;
  EXPECT_LT(certified.kappa, exact.kappa * (1 + 0x1p-8)) << salt;
}

TEST(GramCondition, ManyBlocksAddUpToOneGram) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (uint64_t salt = 0; salt < correlated_salts; ++salt) {
    expect_tall_gram_certified_near_the_exact(res, salt);
  }
}

TEST(GramCondition, AChunkOfNoVectorsIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  try {
    estimate_of<float>(res, summed(weighted_rows, 1), {}, 0.0, {.chunk = 0});
    ADD_FAILURE() << "accepted";
  } catch (const cumlrkt::logic_error& e) {
    EXPECT_STREQ(e.what(), "chunk: expected at least 1 vector, got 0");
  }
}

TEST(GramCondition, ZeroWeightRowsDoNotCount) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  columns c = summed(weighted_rows, 0);
  c.x[(2 * weighted_rows) + last_row] += outlier;
  std::vector<double> w(weighted_rows, 1.0);
  EXPECT_LT(condition_of<float>(res, c, w, 0.0, true), 0x1p24);
  w[last_row] = 0.0;
  EXPECT_GT(condition_of<float>(res, c, w, 0.0, true), 0x1p40);
}

TEST(GramCondition, AConstantColumnWithoutAlphaIsInfinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const columns c{{2, 2, 2, 2, 1, 2, 3, 4}, 4, 2};
  const condition_estimate e = estimate_of<double>(res, c, {}, 0.0, {});
  EXPECT_EQ(e.kappa, std::numeric_limits<double>::infinity());
  EXPECT_EQ(e.path, gram_path::none);
  EXPECT_EQ(e.window, 0);
}

void expect_ols_checked_only_under_float32_qr(int algorithm) {
  const glm_solver ols{glm_family::ols, algorithm};
  const bool qr = algorithm == CUML_LINEAR_MODEL_OLS_QR ||
                  algorithm == CUML_LINEAR_MODEL_OLS_SVD_QR;
  EXPECT_EQ(checks_conditioning<float>(ols, false), qr) << algorithm;
  EXPECT_FALSE(checks_conditioning<double>(ols, false)) << algorithm;
}

TEST(GramCondition, OnlyFloat32QrSvdQrAndRidgeSvdInTheWindowAreChecked) {
  const glm_solver ridge{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_SVD};
  const glm_solver ridge_eig{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_EIG};
  for (const int algorithm :
       {CUML_LINEAR_MODEL_OLS_SVD, CUML_LINEAR_MODEL_OLS_EIG,
        CUML_LINEAR_MODEL_OLS_QR, CUML_LINEAR_MODEL_OLS_SVD_QR}) {
    expect_ols_checked_only_under_float32_qr(algorithm);
  }
  EXPECT_TRUE(checks_conditioning<double>(ridge, true));
  EXPECT_FALSE(checks_conditioning<double>(ridge, false));
  EXPECT_FALSE(checks_conditioning<float>(ridge_eig, true));
}

TEST(GramCondition, RefusesAtTheThresholdNamingWhatToChange) {
  const glm_solver svd_qr{glm_family::ols, CUML_LINEAR_MODEL_OLS_SVD_QR};
  const glm_solver ridge{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_SVD};
  EXPECT_EQ(refusal_of(0x1p31, svd_qr, true), "accepted");
  EXPECT_EQ(refusal_of(0x1p32, svd_qr, true),
            "X: the svd-qr solve's X has u*kappa 0.00390625 >= "
            "2^-8; X may have collinear columns");
  EXPECT_EQ(refusal_of(0x1p34, ridge, false), "accepted");
  EXPECT_EQ(refusal_of(0x1p35, ridge, false),
            "X: the svd solve's regularised Gram matrix has u*kappa "
            "3.814697265625e-06 >= 2^-18; X is ill-conditioned for this "
            "alpha (a larger alpha may fit)");
  EXPECT_EQ(refusal_of(std::numeric_limits<double>::infinity(), ridge, true),
            "X: the svd solve's regularised Gram matrix has u*kappa inf >= "
            "2^-8; X is ill-conditioned for this alpha (float64 or a larger "
            "alpha may fit)");
}

}  // namespace
