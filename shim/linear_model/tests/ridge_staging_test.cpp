#include "ridge_staging.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "linear_model_fixture.hpp"

namespace {

using cumlrkt::glm::glm_family;
using cumlrkt::glm::ridge_algorithm;
using cumlrkt::glm::ridge_scale_of;
using cumlrkt::glm::staging;
using cumlrkt::glm::workspace_of;
using cumlrkt::test::filled;
using cumlrkt::test::fit_into;
using cumlrkt::test::fitted;
using cumlrkt::test::resources;
using cumlrkt::test::ridge_into;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();

staging planned(std::vector<int> exponent, std::vector<double> largest) {
  staging s;
  s.column_exponent = std::move(exponent);
  s.largest = std::move(largest);
  s.spread.assign(s.largest.size(), 1.0);
  s.y_exponent = 3;
  for (std::size_t j = 0; j < s.largest.size(); ++j) {
    s.kept.push_back(static_cast<int64_t>(j));
  }
  return s;
}

struct staged_case {
  std::vector<int> exponent;
  std::vector<double> largest;
  double alpha;
  int weight_power;
  int64_t n;
  int shared;
  double alpha_staged;
  bool negligible;
};

template <typename T>
void expect_staged(const staged_case& c) {
  const auto r = ridge_scale_of<T>(planned(c.exponent, c.largest), c.n,
                                   {c.alpha, c.weight_power});
  EXPECT_EQ(r.shared, c.shared) << c.alpha;
  EXPECT_EQ(r.alpha, static_cast<T>(c.alpha_staged)) << c.alpha;
  EXPECT_EQ(r.negligible, c.negligible) << c.alpha;
}

template <typename T>
std::string scale_refusal(double alpha) {
  try {
    ridge_scale_of<T>(planned({0}, {1.5}), 10, {alpha, 0});
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

struct routed_case {
  int solver;
  int64_t rows;
  int64_t kept;
  int routed;
};

constexpr int64_t wide_columns = 24000;
constexpr int64_t wide_lwork = 1150000000;

std::string workspace_refusal(int64_t columns, int64_t lwork) {
  try {
    cumlrkt::glm::require_workspace(
        {glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_EIG}, {2, columns}, lwork);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

void expect_least_squares(const resources& res, const filled<float>& x,
                          const filled<float>& y, std::pair<int, int> codes,
                          double alpha) {
  fitted ours(res);
  fitted theirs(res);
  ASSERT_EQ(
      ridge_into(res, ours, x.view(), y.view(), nullptr, alpha, 1, codes.first),
      "accepted");
  ASSERT_EQ(fit_into(res, theirs, x.view(), y.view(), nullptr, 1, codes.second),
            "accepted");
  EXPECT_EQ(ours.out.read<float>(0), theirs.out.read<float>(0)) << alpha;
  EXPECT_EQ(ours.intercept, theirs.intercept) << alpha;
}

TEST(RidgeStaging, SharesOneExponentAndScalesAlphaByItsSquare) {
  const std::vector<staged_case> f32{
      {{-3, 2}, {12.0, 0.5}, 1.0, 0, 100, 13, 0x1p26, false},
      {{-3, 2}, {12.0, 0.5}, 0x1p20, 0, 100, 6, 0x1p32, false},
      {{-3, 2}, {12.0, 0.5}, 0x1p21, 0, 100, 5, 0x1p31, false},
      {{-3, 2}, {12.0, 0.5}, 3.0, -4, 100, 13, 0x3p22, false},
      {{-3, 2}, {12.0, 0.5}, 0x1p-190, 0, 100, 13, 0.0, true},
  };
  for (const staged_case& c : f32) {
    expect_staged<float>(c);
  }
  const std::vector<staged_case> f64{
      {{0}, {1.5}, 1e-300, 0, 10, 64, std::ldexp(1e-300, 128), true},
      {{18}, {0x1p1000}, 1.0, 0, 10, 18, 0x1p36, true},
  };
  for (const staged_case& c : f64) {
    expect_staged<double>(c);
  }
}

TEST(RidgeStaging, AnAlphaFarPastTheSpreadKeepsTheWidestColumnAtTheFloor) {
  expect_staged<float>({{0}, {1.5}, 0x1p100, 0, 10, 8, 0x1p116, false});
  expect_staged<double>(
      {{0}, {1.5}, 0x1p500, -2, 10, 32, std::ldexp(0x1p500, 62), false});
  EXPECT_EQ(scale_refusal<float>(0x1p120),
            "alpha: expected at most 5.192296549049818e+33 for this X in "
            "float32, got 1.329227995784916e+36");
  EXPECT_EQ(scale_refusal<double>(0x1p1000),
            "alpha: expected at most 9.745314011399998e+288 for this X in "
            "float64, got 1.0715086071862673e+301");
}

TEST(RidgeStaging, AnAlphaUnderRoundingAgainstTheWidestSpreadIsNegligible) {
  expect_staged<float>({{0}, {1.5}, 0x1p-24, 0, 10, 16, 0x1p8, true});
  expect_staged<float>({{0}, {1.5}, 0x1p-23, 0, 10, 16, 0x1p9, false});
  expect_staged<double>({{0}, {1.5}, 0x1p-53, 0, 10, 64, 0x1p75, true});
  expect_staged<double>({{0}, {1.5}, 0x1p-52, 0, 10, 64, 0x1p76, false});
}

TEST(RidgeStaging, RoutesTheCasesRidgeFitCannotTake) {
  const std::vector<routed_case> routes{
      {CUML_LINEAR_MODEL_RIDGE_EIG, 10, 3, CUML_LINEAR_MODEL_RIDGE_EIG},
      {CUML_LINEAR_MODEL_RIDGE_SVD, 10, 3, CUML_LINEAR_MODEL_RIDGE_SVD},
      {CUML_LINEAR_MODEL_RIDGE_SVD, 3, 3, CUML_LINEAR_MODEL_RIDGE_SVD},
      {CUML_LINEAR_MODEL_RIDGE_SVD, 2, 3, CUML_LINEAR_MODEL_RIDGE_EIG},
      {CUML_LINEAR_MODEL_RIDGE_EIG, 10, 1, CUML_LINEAR_MODEL_RIDGE_SVD},
      {CUML_LINEAR_MODEL_RIDGE_SVD, 1, 1, CUML_LINEAR_MODEL_RIDGE_SVD},
  };
  for (const routed_case& c : routes) {
    EXPECT_EQ(ridge_algorithm(c.solver, c.rows, c.kept), c.routed)
        << c.solver << " " << c.rows << " " << c.kept;
  }
  EXPECT_EQ(
      workspace_of({glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_SVD}, {10, 3}, 5),
      53);
  EXPECT_EQ(
      workspace_of({glm_family::ridge, CUML_LINEAR_MODEL_RIDGE_EIG}, {10, 3}, 5),
      62);
  EXPECT_EQ(workspace_refusal(wide_columns, wide_lwork),
            "X: 24000 columns need a 2302120000-element eig workspace, more "
            "than 2147483647 elements");
}

TEST(Ridge, AlphaThatUnderflowsIsTheLeastSquaresFitBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, {1, 0, 2, 0, 3, 1, 4, 1, 0, 2, 2, 5, 3, 1, 1},
                        {5, 3});
  const filled<float> y(res, {1, 2, 3, 4, 5}, {5});
  for (const auto [solver, algorithm] :
       {std::pair{CUML_LINEAR_MODEL_RIDGE_EIG, CUML_LINEAR_MODEL_OLS_EIG},
        std::pair{CUML_LINEAR_MODEL_RIDGE_SVD, CUML_LINEAR_MODEL_OLS_SVD_QR}}) {
    for (const double alpha : {0.0, -0.0, 0x1p-190}) {
      expect_least_squares(res, x, y, {solver, algorithm}, alpha);
    }
  }
}

TEST(RidgeRefusal, CodesAndAlphaAreRefusedAfterTheViews) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {1, 0, 2, 0, 3, 1, 4, 1, 0, 2, 2, 5, 3, 1, 1},
                         {5, 3});
  const filled<double> y(res, {1, 2, 3, 4, 5}, {5});
  const filled<double> y_inf(res, {1, 2, 3, 4, inf}, {5});
  const filled<int32_t> x_int(res, std::vector<int32_t>(15, 1), {5, 3});
  struct row {
    const rr_view* x;
    const rr_view* y;
    double alpha;
    int32_t fit_intercept;
    int32_t solver;
    std::string message;
  };
  const std::vector<row> rows{
      {x_int.view(), y.view(), nan, 1, 7,
       "X: expected float32 or float64, got int32"},
      {x.view(), y.view(), nan, 1, 2, "solver: expected auto, eig or svd"},
      {x.view(), y.view(), nan, 1, -1, "solver: expected auto, eig or svd"},
      {x.view(), y.view(), nan, 2, 1, "fit-intercept: expected #t or #f"},
      {x.view(), y.view(), nan, 1, 1,
       "alpha: expected a finite value at least 0, got nan"},
      {x.view(), y.view(), inf, 1, 1,
       "alpha: expected a finite value at least 0, got inf"},
      {x.view(), y.view(), -inf, 1, 1,
       "alpha: expected a finite value at least 0, got -inf"},
      {x.view(), y.view(), -1.0, 1, 0,
       "alpha: expected a finite value at least 0, got -1"},
      {x.view(), y_inf.view(), -1.0, 1, 0,
       "alpha: expected a finite value at least 0, got -1"},
      {x.view(), y_inf.view(), 1.0, 1, 0,
       "y: element 4: expected a finite value, got inf"},
  };
  for (const row& r : rows) {
    fitted f(res);
    EXPECT_EQ(ridge_into(res, f, r.x, r.y, nullptr, r.alpha, r.fit_intercept,
                         r.solver),
              r.message);
    EXPECT_EQ(cuml_linear_model_last_error_kind(), RR_ERROR_LOGIC) << r.message;
    cumlrkt::test::expect_untouched(f.out);
  }
}

TEST(Ridge, FitsTheOneColumnExampleUnderBothSolvers) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {0, 1, 2, 3}, {4, 1});
  const filled<double> y(res, {1, 3, 5, 7}, {4});
  for (const int32_t solver :
       {CUML_LINEAR_MODEL_RIDGE_SVD, CUML_LINEAR_MODEL_RIDGE_EIG}) {
    fitted f(res);
    ASSERT_EQ(ridge_into(res, f, x.view(), y.view(), nullptr, 1.0, 1, solver),
              "accepted");
    EXPECT_NEAR(f.out.read<double>(0).at(0), 5.0 / 3.0, 0x1p-48) << solver;
    EXPECT_NEAR(f.intercept, 1.5, 0x1p-48) << solver;
  }
}

}  // namespace
