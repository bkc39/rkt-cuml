#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/metrics.h"
#include "metrics_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t n = 4;
constexpr int64_t pool = 64;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr std::size_t long_truth = 1000;
constexpr int up = 600;
constexpr int down = -600;
constexpr int up_error = 250;
constexpr int down_error = -250;
constexpr int up_float = 60;
constexpr int down_float = -60;
constexpr float big_float = 1e19F;
constexpr double tenth = 0.1;
constexpr double third = 1.0 / 3.0;
constexpr double nudge = 1e-3;
constexpr double huge_value = 0x1p600;
constexpr double big_value = 1e200;
constexpr double cancelled_value = 0x1p540;
constexpr double dropped_value = 0x1p1000;
constexpr double half = 0.5;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr float inf_float = std::numeric_limits<float>::infinity();
constexpr std::size_t thirds = 7;
constexpr double weighted_accuracy = 0.75;
constexpr double scaled_correct = 1.5;
constexpr double scaled_total = 2.0;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct refusal {
  std::function<int()> call;
  std::string message;
};

void expect_refusals(const std::vector<refusal>& cases) {
  for (const refusal& c : cases) {
    const int status = c.call();
    EXPECT_EQ(status, RR_ERROR) << c.message;
    EXPECT_EQ(outcome(status), c.message);
    EXPECT_EQ(cuml_metrics_last_error_kind(), RR_ERROR_LOGIC) << c.message;
  }
}

using entry =
    std::function<int(const rr_view*, const rr_view*, const rr_view*)>;

struct shapes {
  explicit shapes(const resources& res)
      : doubles(res.r, RR_DTYPE_FLOAT64, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        longs(res.r, RR_DTYPE_INT64, {pool}) {}
  device_array doubles;
  device_array floats;
  device_array ints;
  device_array longs;
};

rr_view strided(const device_array& a) {
  rr_view v = shaped(a.view, {n});
  v.strides[0] = 2;
  return v;
}

struct scored {
  int status;
  double value;
};

struct accuracy_outputs {
  double correct;
  double total;
  std::string status;
};

template <typename T, typename P = T>
scored r2(const resources& res, const std::vector<T>& y,
          const std::vector<P>& p) {
  const auto size = static_cast<int64_t>(y.size());
  const filled<T> truth(res, y, {size});
  const filled<P> pred(res, p, {size});
  double score = 0;
  const int status = cuml_metrics_r2_score(res.handle, truth.view(),
                                           pred.view(), nullptr, &score);
  return {status, score};
}

template <typename T, typename P>
scored mse(const resources& res, const std::vector<T>& y,
           const std::vector<P>& p) {
  const auto size = static_cast<int64_t>(y.size());
  const filled<T> truth(res, y, {size});
  const filled<P> pred(res, p, {size});
  double error = 0;
  const int status = cuml_metrics_mean_squared_error(
      res.handle, truth.view(), pred.view(), nullptr, &error);
  return {status, error};
}

template <typename T>
std::vector<T> scaled(const std::vector<T>& xs, int k) {
  std::vector<T> out;
  out.reserve(xs.size());
  for (const T x : xs) {
    out.push_back(std::ldexp(x, k));
  }
  return out;
}

template <typename T>
std::vector<T> nudged(std::vector<T> xs) {
  xs.back() = static_cast<T>(xs.back() + nudge);
  return xs;
}

std::vector<refusal> float_rows(const entry& call, const shapes& s) {
  const rr_view y = shaped(s.doubles.view, {n});
  const rr_view p = shaped(s.floats.view, {n});
  return {
      {[=, &s] { return call(&s.ints.view, &p, nullptr); },
       "y-true: expected float32 or float64, got int32"},
      {[=, &s] {
         const rr_view v = strided(s.doubles);
         return call(&v, &p, nullptr);
       },
       "y-true: expected contiguous elements, got stride 2"},
      {[=, &s] {
         const rr_view v = shaped(s.longs.view, {n});
         return call(&y, &v, nullptr);
       },
       "y-pred: expected float32 or float64, got int64"},
      {[=, &s] {
         const rr_view v = shaped(s.doubles.view, {n, 1});
         return call(&y, &v, nullptr);
       },
       "y-pred: expected rank 1, got rank 2"},
      {[=, &s] {
         const rr_view v = shaped(s.floats.view, {n - 1});
         return call(&y, &v, nullptr);
       },
       "y-pred: expected 4 elements, got 3"},
      {[=, &s] {
         const rr_view v = strided(s.floats);
         return call(&y, &v, nullptr);
       },
       "y-pred: expected contiguous elements, got stride 2"},
      {[=, &s] {
         const rr_view v = shaped(s.doubles.view, {0});
         return call(&v, &v, nullptr);
       },
       "y-true: expected at least 1 element, got 0"},
      {[=, &s] {
         const rr_view v = shaped(s.doubles.view, {past_index});
         return call(&v, &v, nullptr);
       },
       "y-true: extent 2147483648 does not fit the index type"},
  };
}

std::vector<refusal> label_rows(const entry& call, const shapes& s) {
  const rr_view y = shaped(s.ints.view, {n});
  const rr_view p = shaped(s.longs.view, {n});
  return {
      {[=, &s] {
         const rr_view v = shaped(s.floats.view, {n});
         return call(&v, &p, nullptr);
       },
       "y-true: expected int32 or int64, got float32"},
      {[=, &s] {
         const rr_view v = shaped(s.ints.view, {n, 2});
         return call(&v, &p, nullptr);
       },
       "y-true: expected rank 1, got rank 2"},
      {[=, &s] {
         const rr_view v = strided(s.longs);
         return call(&v, &p, nullptr);
       },
       "y-true: expected contiguous elements, got stride 2"},
      {[=, &s] {
         const rr_view v = shaped(s.doubles.view, {n});
         return call(&y, &v, nullptr);
       },
       "y-pred: expected int32 or int64, got float64"},
      {[=, &s] {
         const rr_view v = shaped(s.ints.view, {n - 1});
         return call(&y, &v, nullptr);
       },
       "y-pred: expected 4 elements, got 3"},
      {[=, &s] {
         const rr_view v = shaped(s.ints.view, {0});
         return call(&v, &v, nullptr);
       },
       "y-true: expected at least 1 element, got 0"},
      {[=, &s] {
         const rr_view v = shaped(s.longs.view, {past_index});
         return call(&v, &v, nullptr);
       },
       "y-true: extent 2147483648 does not fit the index type"},
  };
}

std::vector<refusal> weight_rows(const entry& call, const rr_view* y,
                                 const rr_view* p, const shapes& s,
                                 const resources& res) {
  return {
      {[=, &s] {
         const rr_view w = shaped(s.ints.view, {n});
         return call(y, p, &w);
       },
       "sample-weight: expected float32 or float64, got int32"},
      {[=, &s] {
         const rr_view w = shaped(s.doubles.view, {n, 1});
         return call(y, p, &w);
       },
       "sample-weight: expected rank 1, got rank 2"},
      {[=, &s] {
         const rr_view w = shaped(s.doubles.view, {n - 1});
         return call(y, p, &w);
       },
       "sample-weight: expected 4 elements, got 3"},
      {[=, &s] {
         const rr_view w = strided(s.floats);
         return call(y, p, &w);
       },
       "sample-weight: expected contiguous elements, got stride 2"},
      {[=, &res] {
         const filled<double> w(res, {-inf, 1, 1, 1}, {n});
         return call(y, p, w.view());
       },
       "sample-weight: element 0: expected a finite value, got -inf"},
      {[=, &res] {
         const filled<float> w(res, {1, -1, 1, 1}, {n});
         return call(y, p, w.view());
       },
       "sample-weight: element 1: expected at least 0, got -1"},
      {[=, &res] {
         const filled<double> w(res, {0, 0, 0, 0}, {n});
         return call(y, p, w.view());
       },
       "sample-weight: expected a positive sum, got 0"},
      {[=, &res] {
         const filled<double> w(res, {-1, nan, 1, 1}, {n});
         return call(y, p, w.view());
       },
       "sample-weight: element 1: expected a finite value, got nan"},
  };
}

TEST(ScoreRefusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const shapes s(res);
  const filled<double> y(res, {1, 2, 3, 4}, {n});
  const filled<float> p(res, {1, 2, 3, 5}, {n});
  const filled<int32_t> a(res, {0, 1, 2, 3}, {n});
  const filled<int64_t> b(res, {0, 1, 2, 4}, {n});
  const filled<double> nan_y(res, {1, nan, 3, 4}, {n});
  const filled<float> inf_p(res, {1, 2, 3, inf_float}, {n});
  double score = 0;
  double correct = 0;
  double total = 0;
  const entry r2_call = [&](const rr_view* t, const rr_view* q,
                            const rr_view* w) {
    return cuml_metrics_r2_score(res.handle, t, q, w, &score);
  };
  const entry mse_call = [&](const rr_view* t, const rr_view* q,
                             const rr_view* w) {
    return cuml_metrics_mean_squared_error(res.handle, t, q, w, &score);
  };
  const entry accuracy_call = [&](const rr_view* t, const rr_view* q,
                                  const rr_view* w) {
    return cuml_metrics_accuracy_score(res.handle, t, q, w, &correct, &total);
  };
  for (const entry& call : {r2_call, mse_call}) {
    expect_refusals(float_rows(call, s));
    expect_refusals(weight_rows(call, y.view(), p.view(), s, res));
    expect_refusals({
        {[&] { return call(nan_y.view(), p.view(), nullptr); },
         "y-true: element 1: expected a finite value, got nan"},
        {[&] { return call(y.view(), inf_p.view(), nullptr); },
         "y-pred: element 3: expected a finite value, got inf"},
        {[&] { return call(nan_y.view(), inf_p.view(), nullptr); },
         "y-true: element 1: expected a finite value, got nan"},
    });
  }
  expect_refusals(label_rows(accuracy_call, s));
  expect_refusals(weight_rows(accuracy_call, a.view(), b.view(), s, res));
  expect_refusals({
      {[&] {
         return cuml_metrics_r2_score(res.handle, y.view(), p.view(), nullptr,
                                      nullptr);
       },
       "score is NULL"},
      {[&] {
         return cuml_metrics_mean_squared_error(res.handle, y.view(), p.view(),
                                                nullptr, nullptr);
       },
       "error is NULL"},
      {[&] {
         return cuml_metrics_accuracy_score(res.handle, a.view(), b.view(),
                                            nullptr, nullptr, &total);
       },
       "correct is NULL"},
      {[&] {
         return cuml_metrics_accuracy_score(res.handle, a.view(), b.view(),
                                            nullptr, &correct, nullptr);
       },
       "total is NULL"},
  });
}

TEST(ScoreRefusals, MatrixShapes) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const shapes s(res);
  const filled<double> y(res, {1, 2, 3, 4, 5, 6, 7, 8}, {n, 2});
  const filled<float> p(res, {1, 2, 3, 4, 5, 6, 7, 9}, {n, 2});
  const filled<float> inf_y(res, {1, 2, inf_float, 4, 5, 6, 7, 8}, {n, 2});
  double error = 0;
  const auto call = [&](const rr_view* t, const rr_view* q) {
    return cuml_metrics_mean_squared_error(res.handle, t, q, nullptr, &error);
  };
  const rr_view col_major = shaped(s.doubles.view, {n, 2}, true);
  const rr_view vector = shaped(s.doubles.view, {n});
  const rr_view few_rows = shaped(s.doubles.view, {n - 1, 2});
  const rr_view wide = shaped(s.doubles.view, {n, 3});
  const rr_view no_rows = shaped(s.doubles.view, {0, 2});
  const rr_view no_columns = shaped(s.doubles.view, {n, 0});
  const rr_view past = shaped(s.doubles.view, {past_index, 2});
  const rr_view cube = shaped(s.doubles.view, {n, 2, 2});
  const rr_view int_cube = shaped(s.ints.view, {n, 2, 2});
  expect_refusals({
      {[&] { return call(&cube, &cube); },
       "y-true: expected rank 1 or 2, got rank 3"},
      {[&] { return call(&int_cube, &cube); },
       "y-true: expected float32 or float64, got int32"},
      {[&] { return call(&col_major, p.view()); },
       "y-true: expected row-major, got col-major 4x2"},
      {[&] { return call(y.view(), &vector); },
       "y-pred: expected rank 2, got rank 1"},
      {[&] { return call(y.view(), &few_rows); },
       "y-pred: expected 4 rows, got 3"},
      {[&] { return call(y.view(), &wide); },
       "y-pred: expected 2 columns, got 3"},
      {[&] { return call(y.view(), &col_major); },
       "y-pred: expected row-major, got col-major 4x2"},
      {[&] { return call(&no_rows, &no_rows); },
       "y-true: expected at least 1 row, got 0"},
      {[&] { return call(&no_columns, &no_columns); },
       "y-true: expected at least 1 column, got 0"},
      {[&] { return call(inf_y.view(), p.view()); },
       "y-true: row 1, column 0: expected a finite value, got inf"},
      {[&] { return call(&past, &past); },
       "y-true: extent 2147483648 does not fit the index type"},
  });
}

template <typename T>
void expect_constant_scores(const resources& res, const std::vector<T>& y) {
  EXPECT_EQ(r2(res, y, y).value, 1.0);
  EXPECT_EQ(r2(res, y, nudged(y)).value, 0.0);
}

TEST(Scores, ConstantTruthScoresOneOrZero) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<std::vector<double>> constants{
      {2, 2, 2},
      std::vector<double>(3, tenth),
      std::vector<double>(long_truth, tenth),
      std::vector<double>(thirds, third),
      {1}};
  for (const std::vector<double>& y : constants) {
    expect_constant_scores(res, y);
    expect_constant_scores(res, std::vector<float>(y.begin(), y.end()));
  }
  const filled<double> y(res, {5, 1, 5}, {3});
  const filled<double> w(res, {1, 0, 2}, {3});
  const filled<double> hit(res, {5, 9, 5}, {3});
  const filled<double> miss(res, {5, 1, 6}, {3});
  double score = -1;
  ASSERT_EQ(outcome(cuml_metrics_r2_score(res.handle, y.view(), hit.view(),
                                          w.view(), &score)),
            "accepted");
  EXPECT_EQ(score, 1.0);
  ASSERT_EQ(outcome(cuml_metrics_r2_score(res.handle, y.view(), miss.view(),
                                          w.view(), &score)),
            "accepted");
  EXPECT_EQ(score, 0.0);
}

TEST(Scores, ErrorOverflowsOnlyPastTheFlonumRange) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const scored narrow = mse<float, float>(res, {big_float, 0}, {-big_float, 0});
  ASSERT_EQ(outcome(narrow.status), "accepted");
  const double twice = 2 * static_cast<double>(big_float);
  EXPECT_EQ(narrow.value, twice * twice / 2);
  const scored wide = mse<double, double>(res, {huge_value, 0}, {0, 0});
  ASSERT_EQ(outcome(wide.status), "accepted");
  EXPECT_EQ(wide.value, inf);
}

const std::vector<double> scaling_truth{3, -0.5, 2, 7, 1e-3, 4};
const std::vector<double> scaling_pred{2.5, 0, 2, 8, 2e-3, 3};

TEST(Scores, PowerOfTwoScalingIsExact) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<double>& y = scaling_truth;
  const std::vector<double>& p = scaling_pred;
  const double base = r2(res, y, p).value;
  const double error = mse(res, y, p).value;
  for (const int k : {down, up}) {
    EXPECT_EQ(r2(res, scaled(y, k), scaled(p, k)).value, base) << k;
  }
  for (const int k : {down_error, up_error}) {
    EXPECT_EQ(mse(res, scaled(y, k), scaled(p, k)).value,
              std::ldexp(error, 2 * k))
        << k;
  }
}

TEST(Scores, PowerOfTwoScalingIsExactInFloat32) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<float> fy(scaling_truth.begin(), scaling_truth.end());
  const std::vector<float> fp(scaling_pred.begin(), scaling_pred.end());
  const double fbase = r2(res, fy, fp).value;
  const double ferror = mse(res, fy, fp).value;
  for (const int k : {down_float, up_float}) {
    EXPECT_EQ(r2(res, scaled(fy, k), scaled(fp, k)).value, fbase) << k;
    EXPECT_EQ(mse(res, scaled(fy, k), scaled(fp, k)).value,
              std::ldexp(ferror, 2 * k))
        << k;
  }
}

accuracy_outputs accuracy(const resources& res, const rr_view* weights) {
  const filled<int32_t> a(res, {0, 1, 2, 3}, {n});
  const filled<int64_t> b(res, {0, 2, 1, 3}, {n});
  accuracy_outputs found{-1, -1, ""};
  found.status = outcome(cuml_metrics_accuracy_score(
      res.handle, a.view(), b.view(), weights, &found.correct, &found.total));
  return found;
}

TEST(Scores, AccuracyReturnsTheMatchCountAndN) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const accuracy_outputs found = accuracy(res, nullptr);
  ASSERT_EQ(found.status, "accepted");
  EXPECT_EQ(found.correct, 2.0);
  EXPECT_EQ(found.total, 4.0);
}

TEST(Scores, WeightedAccuracyReturnsTheScaledMatchedWeightAndTotal) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> w(res, {1, 1, 1, 5}, {n});
  const accuracy_outputs found = accuracy(res, w.view());
  ASSERT_EQ(found.status, "accepted");
  EXPECT_EQ(found.correct, scaled_correct);
  EXPECT_EQ(found.total, scaled_total);
  EXPECT_EQ(found.correct / found.total, weighted_accuracy);
}

TEST(Scores, SmallResidualsBesideLargeValuesKeepTheirBits) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const double big : {big_value, cancelled_value}) {
    const scored error = mse<double, double>(res, {big, 1}, {big, 2});
    EXPECT_EQ(error.value, half) << big;
  }
}

TEST(Scores, ALargeValueBehindAZeroWeightChangesNothing) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> y(res, {dropped_value, 1, 2}, {3});
  const filled<double> p(res, {dropped_value, 1, 3}, {3});
  const filled<double> w(res, {0, 1, 1}, {3});
  double score = 0;
  ASSERT_EQ(outcome(cuml_metrics_r2_score(res.handle, y.view(), p.view(),
                                          w.view(), &score)),
            "accepted");
  EXPECT_EQ(score, -1.0);
  ASSERT_EQ(outcome(cuml_metrics_mean_squared_error(
                res.handle, y.view(), p.view(), w.view(), &score)),
            "accepted");
  EXPECT_EQ(score, half);
}

}  // namespace
}  // namespace cumlrkt::test
