#include "logistic_checks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "linear_model_fixture.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/c_api.h"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double above_one = 1.5;
constexpr double default_tol = 1e-4;

template <typename T>
std::string refusal_of(const resources& res, const std::vector<T>& values,
                       void (*check)(const T*, int64_t,
                                     rmm::cuda_stream_view)) {
  const filled<T> v(res, values, {static_cast<int64_t>(values.size())});
  const auto stream =
      raft::resource::get_cuda_stream(raftrkt::handle_of(res.handle));
  try {
    check(static_cast<const T*>(v.array.view.data),
          static_cast<int64_t>(values.size()), stream);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(LogisticSolution, NonFiniteParametersAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  struct row {
    std::vector<double> values;
    std::string message;
  };
  const std::vector<row> rows{
      {{1, -2, 0.5}, "accepted"},
      {{1, nan, 0.5}, "X: the qn solve gave a non-finite coefficient"},
      {{1, -2, -inf}, "X: the qn solve gave a non-finite coefficient"},
  };
  for (const row& r : rows) {
    EXPECT_EQ(
        refusal_of<double>(res, r.values,
                           cumlrkt::glm::logistic::require_finite_solution),
        r.message);
  }
}

TEST(LogisticSolution, CoefficientsOutsideTheDtypeAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<float> finite{1, -2};
  const std::vector<float> overflowed{1,
                                      std::numeric_limits<float>::infinity()};
  EXPECT_EQ(refusal_of<float>(res, finite,
                              cumlrkt::glm::logistic::require_coef_in_range),
            "accepted");
  EXPECT_EQ(refusal_of<float>(res, overflowed,
                              cumlrkt::glm::logistic::require_coef_in_range),
            "X: a coefficient exceeds the float32 range");
}

std::string bounds_refusal(const cumlrkt::glm::logistic::fit_options& o) {
  try {
    cumlrkt::glm::logistic::require_fit_bounds(o);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(LogisticOptions, EachOutOfRangeOptionIsRefused) {
  using cumlrkt::glm::logistic::fit_options;
  const fit_options base{.penalty = 2,
                         .C = 1.0,
                         .l1_ratio = 0.0,
                         .tol = default_tol,
                         .max_iter = 1000,
                         .linesearch_max_iter = 50,
                         .lbfgs_memory = 5,
                         .fit_intercept = true,
                         .normalized = true,
                         .class_mode = 0};
  struct row {
    fit_options options;
    std::string message;
  };
  const auto with = [&](auto edit) {
    fit_options o = base;
    edit(o);
    return o;
  };
  const std::vector<row> rows{
      {base, "accepted"},
      {with([](fit_options& o) { o.lbfgs_memory = 0; }),
       "lbfgs-memory: expected at least 1, got 0"},
      {with([](fit_options& o) { o.lbfgs_memory = -1; }),
       "lbfgs-memory: expected at least 1, got -1"},
      {with([](fit_options& o) { o.max_iter = 0; }),
       "max-iter: expected at least 1, got 0"},
      {with([](fit_options& o) { o.linesearch_max_iter = 0; }),
       "linesearch-max-iter: expected at least 1, got 0"},
      {with([](fit_options& o) { o.C = 0; }),
       "C: expected a positive finite real, got 0"},
      {with([](fit_options& o) { o.C = nan; }),
       "C: expected a positive finite real, got nan"},
      {with([](fit_options& o) { o.C = inf; }),
       "C: expected a positive finite real, got inf"},
      {with([](fit_options& o) { o.l1_ratio = nan; }),
       "l1-ratio: expected a real from 0 to 1, got nan"},
      {with([](fit_options& o) { o.l1_ratio = above_one; }),
       "l1-ratio: expected a real from 0 to 1, got 1.5"},
      {with([](fit_options& o) { o.tol = nan; }),
       "tol: expected a finite value of at least 0, got nan"},
      {with([](fit_options& o) { o.tol = -1; }),
       "tol: expected a finite value of at least 0, got -1"},
  };
  for (const row& r : rows) {
    EXPECT_EQ(bounds_refusal(r.options), r.message);
  }
}

TEST(LogisticOptions, TheFitEntryRefusesThemBeforeTheSolver) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {0, 1, 1, 0, 2, 1, 3, 0, 4, 1, 5, 0}, {6, 2});
  const filled<int32_t> y(res, {0, 1, 0, 1, 0, 1}, {6});
  struct row {
    int32_t max_iter;
    int32_t linesearch_max_iter;
    int32_t lbfgs_memory;
    std::string message;
  };
  const std::vector<row> rows{
      {1000, 50, 0, "lbfgs-memory: expected at least 1, got 0"},
      {1000, 50, -3, "lbfgs-memory: expected at least 1, got -3"},
      {0, 50, 5, "max-iter: expected at least 1, got 0"},
      {1000, 0, 5, "linesearch-max-iter: expected at least 1, got 0"},
  };
  for (const row& r : rows) {
    cumlrkt::test::fitted f(res);
    int32_t n_iter = -1;
    EXPECT_EQ(cumlrkt::test::outcome(cuml_linear_model_logistic_regression_fit(
                  res.handle, x.view(), y.view(), nullptr, nullptr, &f.out.r,
                  CUML_LINEAR_MODEL_PENALTY_L2, 1.0, 0.0, default_tol,
                  r.max_iter, r.linesearch_max_iter, r.lbfgs_memory, 1, 1,
                  CUML_LINEAR_MODEL_CLASS_WEIGHT_NONE, &n_iter)),
              r.message);
    EXPECT_EQ(cuml_linear_model_last_error_kind(), RR_ERROR_LOGIC) << r.message;
    cumlrkt::test::expect_untouched(f.out);
  }
}

}  // namespace
