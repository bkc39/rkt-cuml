#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "cumlrkt/svm.h"
#include "raftrkt/c_api.h"
#include "svm_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 8;
constexpr int64_t cols = 2;
constexpr int64_t quiet_peak = int64_t{64} << 10;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr int64_t forged_side = 46341;

struct fit_case {
  std::vector<double> x;
  std::vector<int32_t> codes;
  std::optional<std::vector<double>> weight;
  int32_t kernel = CUML_SVM_KERNEL_RBF;
  int32_t gamma_mode = CUML_SVM_GAMMA_SCALE;
  double gamma = 0.0;
  double coef0 = 0.0;
  double C = 1.0;
  double tol = 1e-3;
};

std::vector<double> grid() {
  std::vector<double> x;
  for (int64_t i = 0; i < rows; ++i) {
    x.push_back(static_cast<double>(i % 4));
    x.push_back(static_cast<double>((i * 3) % 5));
  }
  return x;
}

std::vector<int32_t> alternating() {
  std::vector<int32_t> codes;
  for (int64_t i = 0; i < rows; ++i) {
    codes.push_back(static_cast<int32_t>(i % 2));
  }
  return codes;
}

template <typename T>
std::vector<T> narrowed(const std::vector<double>& v) {
  return {v.begin(), v.end()};
}

template <typename T>
std::string fitted(const resources& res, const fit_case& c) {
  const filled<T> x(res, narrowed<T>(c.x), {rows, cols});
  const filled<int32_t> codes(res, c.codes, {rows});
  std::optional<filled<T>> w;
  if (c.weight) {
    w.emplace(res, narrowed<T>(*c.weight), std::vector<int64_t>{rows});
  }
  record out(res);
  double gamma = 0.0;
  int32_t n_iter = 0;
  int32_t negative = 0;
  int32_t positive = 0;
  int32_t empty = 2;
  const int status = cuml_svm_svc_fit(
      res.handle, x.view(), codes.view(), w ? w->view() : nullptr, &out.r,
      c.kernel, 3, c.gamma_mode, c.gamma, c.coef0, c.C, c.tol, 1024.0, -1, 1000,
      &gamma, &n_iter, &negative, &positive, &empty);
  if (status != RR_OK || empty >= 0) {
    expect_untouched(out);
  }
  if (status == RR_OK && empty >= 0) {
    return "empty class " + std::to_string(empty);
  }
  return outcome(status);
}

fit_case base() {
  return {grid(), alternating(), std::nullopt};
}

fit_case with_x(double v, std::size_t at) {
  fit_case c = base();
  c.x[at] = v;
  return c;
}

fit_case with_weight(std::vector<double> w) {
  fit_case c = base();
  c.weight = std::move(w);
  return c;
}

fit_case with_settings(double C, double tol, double coef0) {
  fit_case c = base();
  c.C = C;
  c.tol = tol;
  c.coef0 = coef0;
  return c;
}

fit_case scaled_x(double by) {
  fit_case c = base();
  for (double& v : c.x) {
    v *= by;
  }
  return c;
}

struct refusal {
  fit_case c;
  std::string prefix;
};

std::vector<refusal> refusals() {
  std::vector<double> ones(rows, 1.0);
  std::vector<double> negative = ones;
  negative[3] = -1.0;
  std::vector<double> lopsided = ones;
  std::vector<double> unweighted_first = ones;
  for (int64_t i = 0; i < rows; i += 2) {
    lopsided[i + 1] = 0.0;
    unweighted_first[i] = 0.0;
  }
  std::vector<double> undefined = ones;
  undefined[0] = nan;
  fit_case bad_code = base();
  bad_code.codes[0] = 2;
  fit_case constant = base();
  constant.x.assign(rows * cols, 1.5);
  fit_case wide = base();
  wide.gamma_mode = CUML_SVM_GAMMA_NUMBER;
  wide.gamma = 1e30;
  fit_case unbounded_gamma = base();
  unbounded_gamma.gamma_mode = CUML_SVM_GAMMA_NUMBER;
  unbounded_gamma.gamma = inf;
  return {
      {with_settings(inf, 1e-3, 0.0), "C: expected a positive value finite in"},
      {with_settings(0.0, 1e-3, 0.0), "C: expected a positive value finite in"},
      {with_settings(1.0, 0.0, 0.0),
       "tol: expected a positive value finite in"},
      {with_settings(1.0, 1e-3, nan), "coef0: expected a value finite in"},
      {unbounded_gamma, "gamma: expected a value finite in"},
      {bad_code, "y: row 0: expected a class code 0 or 1"},
      {with_weight(undefined), "sample-weight: row 0: expected a finite value"},
      {with_weight(negative), "sample-weight: negative weight -1 at row 3"},
      {with_weight(lopsided), "empty class 1"},
      {with_weight(unweighted_first), "empty class 0"},
      {with_weight(std::vector<double>(rows, 0.0)), "empty class 0"},
      {with_x(nan, 5), "X: row 2, column 1: expected a finite value"},
      {constant, "gamma 'scale: X has zero variance; pass a numeric #:gamma"},
      {wide, "the kernel can overflow (bound "},
  };
}

template <typename T>
void expect_refused_before_the_fit() {
  const allocation_meter meter;
  const resources res;
  for (const refusal& r : refusals()) {
    const int64_t before = meter.peak();
    const std::string got = fitted<T>(res, r.c);
    EXPECT_EQ(got.rfind(r.prefix, 0), 0U) << got;
    EXPECT_LT(meter.peak() - before, quiet_peak) << r.prefix;
  }
}

TEST(svc_refusal, F_refusals_come_before_the_fit_float) {
  CUMLRKT_REQUIRE_GPU();
  expect_refused_before_the_fit<float>();
}

TEST(svc_refusal, F_refusals_come_before_the_fit_double) {
  CUMLRKT_REQUIRE_GPU();
  expect_refused_before_the_fit<double>();
}

TEST(svc_refusal, F_float32_refuses_what_narrows_to_zero_or_infinity) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  EXPECT_EQ(fitted<float>(res, with_settings(1e39, 1e-3, 0.0)),
            "C: expected a positive value finite in float32, got 1e+39");
  EXPECT_EQ(fitted<float>(res, with_settings(1e-50, 1e-3, 0.0)),
            "C: expected a positive value finite in float32, got 1e-50");
  EXPECT_EQ(fitted<float>(res, with_settings(1.0, 1e-3, 1e39)),
            "coef0: expected a value finite in float32, got 1e+39");
  const std::string got = fitted<float>(res, scaled_x(1e-20));
  EXPECT_EQ(got.rfind("gamma: expected a value finite in float32, got ", 0), 0U)
      << got;
  EXPECT_EQ(fitted<double>(res, with_settings(1e39, 1e-3, 0.0)), "accepted");
  EXPECT_EQ(fitted<double>(res, scaled_x(1e-20)), "accepted");
}

TEST(svc_refusal, F_a_class_whose_penalties_narrow_to_zero_is_empty) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<double> faint(rows, 1.0);
  for (int64_t i = 1; i < rows; i += 2) {
    faint[i] = 1e-45;
  }
  fit_case c = with_weight(faint);
  c.C = 0.1;
  EXPECT_EQ(fitted<float>(res, c), "empty class 1");
  EXPECT_EQ(fitted<double>(res, c), "accepted");
}

TEST(svc_refusal, F_an_overlong_row_is_refused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::string got = fitted<float>(res, with_x(1e19, 2));
  EXPECT_EQ(got.rfind("X: row 1: expected a squared norm of at most", 0), 0U)
      << got;
  EXPECT_NE(got.find("; rescale X"), std::string::npos) << got;
}

TEST(svc_refusal, F_an_index_overflow_is_refused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, narrowed<float>(grid()), {rows, cols});
  const filled<int32_t> codes(res, alternating(), {rows});
  const rr_view forged = shaped(*x.view(), {forged_side, forged_side});
  record out(res);
  double gamma = 0.0;
  int32_t n_iter = 0;
  int32_t negative = 0;
  int32_t positive = 0;
  int32_t empty = 2;
  const int status = cuml_svm_svc_fit(
      res.handle, &forged, codes.view(), nullptr, &out.r, CUML_SVM_KERNEL_RBF,
      3, CUML_SVM_GAMMA_SCALE, 0.0, 0.0, 1.0, 1e-3, 1024.0, -1, 1000, &gamma,
      &n_iter, &negative, &positive, &empty);
  expect_untouched(out);
  EXPECT_EQ(outcome(status),
            "X: 46341x46341 elements do not fit the index type");
}

TEST(svc_refusal, V0c_a_fit_without_support_vectors_is_refused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case loose = base();
  loose.tol = 3.0;
  EXPECT_EQ(fitted<double>(res, loose),
            "the fit has no support vectors; lower #:tol or raise #:max-iter");
  EXPECT_EQ(fitted<double>(res, base()), "accepted");
  EXPECT_EQ(fitted<double>(res, with_weight(std::vector<double>(rows, 1.0))),
            "accepted");
}

}  // namespace
}  // namespace cumlrkt::test
