#include "gram_condition.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::glm::checks_conditioning;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::glm_family;
using cumlrkt::glm::glm_solver;
using cumlrkt::glm::require_conditioned;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

struct columns {
  std::vector<double> x;
  int64_t n;
  int64_t d;
};

template <typename T>
double condition_of(const resources& res, const columns& c,
                    const std::vector<double>& w, double alpha,
                    bool fit_intercept) {
  const std::vector<T> host(c.x.begin(), c.x.end());
  const filled<T> x(res, host, {c.n * c.d});
  const std::vector<T> wh(w.begin(), w.end());
  const filled<T> weights(res, wh.empty() ? std::vector<T>{T{1}} : wh,
                          {wh.empty() ? 1 : c.n});
  return equilibrated_condition<T>(
      raftrkt::handle_of(res.handle), static_cast<const T*>(x.array.view.data),
      w.empty() ? nullptr : static_cast<const T*>(weights.array.view.data), c.n,
      c.d, static_cast<T>(alpha), fit_intercept);
}

double draw(int64_t i, int64_t salt) {
  return static_cast<double>(((i * (37 + (2 * salt))) + (11 * salt)) % 23) -
         11.0;
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

TEST(GramCondition, ZeroWeightRowsDoNotCount) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  columns c = summed(61, 0);
  c.x[(2 * 61) + 60] += 5;
  std::vector<double> w(61, 1.0);
  EXPECT_LT(condition_of<float>(res, c, w, 0.0, true), 0x1p24);
  w[60] = 0.0;
  EXPECT_GT(condition_of<float>(res, c, w, 0.0, true), 0x1p40);
}

TEST(GramCondition, AConstantColumnWithoutAlphaIsInfinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const columns c{{2, 2, 2, 2, 1, 2, 3, 4}, 4, 2};
  EXPECT_EQ(condition_of<double>(res, c, {}, 0.0, true),
            std::numeric_limits<double>::infinity());
}

TEST(GramCondition, OnlyFloat32SvdQrAndRidgeSvdInTheWindowAreChecked) {
  const glm_solver svd_qr{glm_family::ols, CUML_LINEAR_MODEL_OLS_SVD_QR};
  const glm_solver eig{glm_family::ols, CUML_LINEAR_MODEL_OLS_EIG};
  const glm_solver ridge{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_SVD};
  const glm_solver ridge_eig{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_EIG};
  EXPECT_TRUE(checks_conditioning<float>(svd_qr, false));
  EXPECT_FALSE(checks_conditioning<double>(svd_qr, false));
  EXPECT_FALSE(checks_conditioning<float>(eig, false));
  EXPECT_TRUE(checks_conditioning<double>(ridge, true));
  EXPECT_FALSE(checks_conditioning<double>(ridge, false));
  EXPECT_FALSE(checks_conditioning<float>(ridge_eig, true));
}

TEST(GramCondition, RefusesAtTheThresholdNamingWhatToChange) {
  const glm_solver svd_qr{glm_family::ols, CUML_LINEAR_MODEL_OLS_SVD_QR};
  const glm_solver ridge{glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_SVD};
  EXPECT_EQ(refusal_of(0x1p31, svd_qr, true), "accepted");
  EXPECT_EQ(refusal_of(0x1p32, svd_qr, true),
            "X: the svd-qr solve's Gram matrix has u*kappa 0.00390625 >= "
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
