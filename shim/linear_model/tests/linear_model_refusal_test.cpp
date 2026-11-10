#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "linear_model_fixture.hpp"
#include "raftrkt/c_api.h"

namespace {

using cumlrkt::test::expect_untouched;
using cumlrkt::test::filled;
using cumlrkt::test::fit_into;
using cumlrkt::test::fitted;
using cumlrkt::test::predict_into;
using cumlrkt::test::record;
using cumlrkt::test::resources;
using cumlrkt::test::shaped;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();

const std::vector<double> grid{1, 0, 2, 0, 3, 1, 4, 1, 0, 2, 2, 5, 3, 1, 1};
const std::vector<double> targets{1, 2, 3, 4, 5};

double largest_gap(const std::vector<double>& got,
                   const std::vector<double>& want) {
  double gap = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    gap = std::max(gap, std::fabs(got.at(i) - want[i]));
  }
  return gap;
}

std::vector<double> with(std::vector<double> v, std::size_t at, double value) {
  v[at] = value;
  return v;
}

TEST(LinearRegressionRefusal, InputsAreRefusedBeforeAnySolve) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {5, 3});
  const filled<double> y(res, targets, {5});
  const filled<float> y32(res, {1, 2, 3, 4, 5}, {5});
  const filled<double> short_y(res, {1, 2, 3, 4}, {4});
  const filled<double> one_row(res, {1, 2, 3}, {1, 3});
  const filled<double> no_cols(res, {}, {5, 0});
  const filled<double> x_nan(res, with(grid, 10, nan), {5, 3});
  const filled<double> y_inf(res, with(targets, 4, inf), {5});
  const filled<double> w_nan(res, {nan, 1, 1, 1, 1}, {5});
  const filled<double> w_negative(res, {1, -1, 1, 1, 1}, {5});
  const filled<double> w_zero(res, {0, 0, 0, 0, 0}, {5});
  const filled<int32_t> x_int(res, std::vector<int32_t>(15, 1), {5, 3});
  const rr_view col_major = shaped(*x.view(), {5, 3}, true);
  struct row {
    const rr_view* x;
    const rr_view* y;
    const rr_view* w;
    int32_t fit_intercept;
    int32_t algorithm;
    std::string message;
  };
  const std::vector<row> rows{
      {x_int.view(), y.view(), nullptr, 1, 1,
       "X: expected float32 or float64, got int32"},
      {y.view(), y.view(), nullptr, 1, 1, "X: expected rank 2, got rank 1"},
      {&col_major, y.view(), nullptr, 1, 1,
       "X: expected row-major, got col-major 5x3"},
      {x.view(), y32.view(), nullptr, 1, 1, "y: expected float64, got float32"},
      {x.view(), x.view(), nullptr, 1, 1, "y: expected rank 1, got rank 2"},
      {x.view(), short_y.view(), nullptr, 1, 1,
       "y: expected 5 elements, got 4"},
      {x.view(), y.view(), y32.view(), 1, 1,
       "sample-weight: expected float64, got float32"},
      {one_row.view(), short_y.view(), nullptr, 1, 1,
       "y: expected 1 elements, got 4"},
      {no_cols.view(), y.view(), nullptr, 1, 1,
       "X: expected at least 1 column, got 0"},
      {x.view(), y.view(), nullptr, 1, 4,
       "algorithm: expected auto, eig, svd, svd-jacobi, qr or svd-qr"},
      {x.view(), y.view(), nullptr, 2, 1, "fit-intercept: expected #t or #f"},
      {x_nan.view(), y.view(), nullptr, 1, 1,
       "X: row 3, column 1: expected a finite value, got nan"},
      {x.view(), y_inf.view(), nullptr, 1, 1,
       "y: element 4: expected a finite value, got inf"},
      {x.view(), y.view(), w_nan.view(), 1, 1,
       "sample-weight: element 0: expected a finite value, got nan"},
      {x.view(), y.view(), w_negative.view(), 1, 1,
       "sample-weight: element 1: expected at least 0, got -1"},
      {x.view(), y.view(), w_zero.view(), 1, 1,
       "sample-weight: expected a positive sum, got 0"},
  };
  for (const row& r : rows) {
    fitted f(res);
    EXPECT_EQ(fit_into(res, f, r.x, r.y, r.w, r.fit_intercept, r.algorithm),
              r.message);
    EXPECT_EQ(cuml_linear_model_last_error_kind(), RR_ERROR_LOGIC) << r.message;
    expect_untouched(f.out);
  }
  const filled<double> single(res, {1, 2, 3}, {1, 3});
  const filled<double> single_y(res, {1}, {1});
  fitted f(res);
  EXPECT_EQ(fit_into(res, f, single.view(), single_y.view()),
            "X: expected at least 2 rows, got 1");
}

TEST(LinearRegressionRefusal, PredictChecksTheResultItIsGiven) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {5, 3});
  const filled<float> x32(res, std::vector<float>(15, 1.0F), {5, 3});
  const filled<double> coef(res, {1, 2, 3}, {3});
  const filled<double> coef_nan(res, {nan, 2, 3}, {3});
  const filled<double> coef2(res, {1, 2}, {2});
  const filled<double> empty(res, {}, {0});
  const filled<double> no_rows(res, {}, {0, 3});
  const filled<double> x_nan(res, with(grid, 10, nan), {5, 3});
  const filled<float> huge(res, {3e38F, 3e38F, 3e38F, 1, 1, 1}, {2, 3});
  const filled<float> coef32(res, {1, 1, 1}, {3});
  struct row {
    const rr_view* x;
    const rr_view* coef;
    double intercept;
    std::string message;
  };
  const std::vector<row> rows{
      {coef.view(), coef.view(), 0, "X: expected rank 2, got rank 1"},
      {x.view(), x.view(), 0, "coef: expected rank 1, got rank 2"},
      {x.view(), empty.view(), 0, "coef: expected at least 1 element, got 0"},
      {x32.view(), coef.view(), 0, "X: expected float64, got float32"},
      {no_rows.view(), coef.view(), 0, "X: expected at least 1 row, got 0"},
      {x.view(), coef2.view(), 0, "X: expected 2 columns, got 3"},
      {x_nan.view(), coef.view(), 0,
       "X: row 3, column 1: expected a finite value, got nan"},
      {x.view(), coef_nan.view(), 0,
       "coef: element 0: expected a finite value, got nan"},
      {x.view(), coef.view(), inf,
       "intercept: expected a finite value, got inf"},
      {huge.view(), coef32.view(), 0,
       "X: row 0: the prediction exceeds the float32 range"},
  };
  for (const row& r : rows) {
    record out(res);
    EXPECT_EQ(predict_into(res, out, r.x, r.coef, r.intercept), r.message);
    expect_untouched(out);
  }
  record out(res);
  EXPECT_EQ(predict_into(res, out, x.view(), coef.view(), 0.5), "accepted");
  EXPECT_EQ(out.read<double>(0),
            (std::vector<double>{7.5, 9.5, 6.5, 21.5, 8.5}));
}

TEST(LinearRegression, FitsTheDocsExampleExactly) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {0, 1, 1, 1, 2, 1, 3, 1}, {4, 2});
  const filled<double> y(res, {1, 3, 5, 7}, {4});
  for (const int32_t algorithm :
       {CUML_LINEAR_MODEL_OLS_SVD, CUML_LINEAR_MODEL_OLS_EIG,
        CUML_LINEAR_MODEL_OLS_QR, CUML_LINEAR_MODEL_OLS_SVD_QR}) {
    fitted f(res);
    EXPECT_EQ(fit_into(res, f, x.view(), y.view(), nullptr, 1, algorithm),
              "accepted");
    EXPECT_EQ(f.out.read<double>(0), (std::vector<double>{2.0, 0.0}))
        << algorithm;
    EXPECT_EQ(f.intercept, 1.0) << algorithm;
  }
}

TEST(LinearRegression, FitsTheTwoColumnDocsExampleUnderEveryAlgorithm) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {1, 1, 1, 2, 2, 2, 2, 3}, {4, 2});
  const filled<double> y(res, {6, 8, 9, 11}, {4});
  constexpr double close = 0x1p-45;
  for (const int32_t algorithm :
       {CUML_LINEAR_MODEL_OLS_SVD, CUML_LINEAR_MODEL_OLS_EIG,
        CUML_LINEAR_MODEL_OLS_QR, CUML_LINEAR_MODEL_OLS_SVD_QR}) {
    fitted f(res);
    EXPECT_EQ(fit_into(res, f, x.view(), y.view(), nullptr, 1, algorithm),
              "accepted");
    std::vector<double> got = f.out.read<double>(0);
    got.push_back(f.intercept);
    EXPECT_LE(largest_gap(got, {1.0, 2.0, 3.0}), close) << algorithm;
  }
}

using cumlrkt::test::cd_settings;
using cumlrkt::test::elastic_net_into;

struct cd_refusal {
  cd_settings settings;
  const char* message;
};

const std::vector<cd_refusal> cd_refusals{
    {{.selection = 2}, "selection: expected cyclic or random"},
    {{.alpha = -1, .selection = 2}, "selection: expected cyclic or random"},
    {{.fit_intercept = 2}, "fit-intercept: expected #t or #f"},
    {{.alpha = -1, .fit_intercept = 2}, "fit-intercept: expected #t or #f"},
    {{.alpha = -1}, "alpha: expected a finite value at least 0, got -1"},
    {{.alpha = nan}, "alpha: expected a finite value at least 0, got nan"},
    {{.alpha = inf}, "alpha: expected a finite value at least 0, got inf"},
    {{.alpha = -1, .l1_ratio = 2},
     "alpha: expected a finite value at least 0, got -1"},
    {{.l1_ratio = -0.5},
     "l1-ratio: expected a finite value from 0 to 1, got -0.5"},
    {{.l1_ratio = 1.5},
     "l1-ratio: expected a finite value from 0 to 1, got 1.5"},
    {{.l1_ratio = nan},
     "l1-ratio: expected a finite value from 0 to 1, got nan"},
    {{.l1_ratio = 2, .max_iter = 0},
     "l1-ratio: expected a finite value from 0 to 1, got 2"},
    {{.max_iter = 0}, "max-iter: expected an exact integer from 1 to 2147483647, got 0"},
    {{.max_iter = -1}, "max-iter: expected an exact integer from 1 to 2147483647, got -1"},
    {{.max_iter = 0, .tol = -1}, "max-iter: expected an exact integer from 1 to 2147483647, got 0"},
    {{.tol = -1}, "tol: expected a finite value of at least 0, got -1"},
    {{.tol = nan}, "tol: expected a finite value of at least 0, got nan"},
    {{.tol = inf}, "tol: expected a finite value of at least 0, got inf"},
};

void expect_code_refused(const resources& res, const rr_view* x,
                         const rr_view* y, const cd_refusal& r) {
  fitted f(res);
  EXPECT_EQ(elastic_net_into(res, f, x, y, r.settings), r.message);
  EXPECT_EQ(cuml_linear_model_last_error_kind(), RR_ERROR_LOGIC) << r.message;
  expect_untouched(f.out);
}

TEST(ElasticNetRefusal, CodesAreRefusedAfterTheViewsAndBeforeTheValues) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {5, 3});
  const filled<double> y(res, targets, {5});
  const filled<double> x_nan(res, with(grid, 10, nan), {5, 3});
  const filled<float> y32(res, {1, 2, 3, 4, 5}, {5});
  for (const cd_refusal& r : cd_refusals) {
    expect_code_refused(res, x.view(), y.view(), r);
    expect_code_refused(res, x_nan.view(), y.view(), r);
    fitted v(res);
    EXPECT_EQ(elastic_net_into(res, v, x.view(), y32.view(), r.settings),
              "y: expected float64, got float32");
  }
}

TEST(ElasticNetRefusal, ZeroToleranceAndZeroAlphaAreAccepted) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {5, 3});
  const filled<double> y(res, targets, {5});
  fitted f(res);
  EXPECT_EQ(elastic_net_into(res, f, x.view(), y.view(),
                             {.alpha = 0, .max_iter = 3, .tol = 0}),
            "accepted");
}

}  // namespace
