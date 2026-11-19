#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "linear_model_fixture.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "logistic_resolve.hpp"
#include "logistic_solve.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::matrix;
using cumlrkt::glm::row_selection;
using cumlrkt::glm::to_device;
using cumlrkt::glm::logistic::attempt;
using cumlrkt::glm::logistic::column_stats;
using cumlrkt::glm::logistic::fit_options;
using cumlrkt::glm::logistic::penalties;
using cumlrkt::glm::logistic::qn_shape;
using cumlrkt::glm::logistic::qn_solution;
using cumlrkt::glm::logistic::qn_solver;
using cumlrkt::glm::logistic::rescue;
using cumlrkt::glm::logistic::rescued;
using cumlrkt::glm::logistic::solve_check;
using cumlrkt::glm::logistic::staged_fit;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t rows = 4;
constexpr int64_t params = 2;
constexpr int first_iterations = 3;
constexpr int max_iter = 100;
constexpr int linesearch_max_iter = 50;
constexpr int lbfgs_memory = 5;
constexpr double tol = 1e-4;
constexpr double mean = 1.5;
constexpr double largest = 3.0;
constexpr double first_worst = 1.0;
constexpr double first_bound = 1e-3;

struct outcome {
  bool rescued;
  int n_iter;
  std::vector<std::vector<double>> starts;
  std::vector<int> budgets;
};

outcome rescued_with(bool cold_spends_the_budget) {
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  const auto stream = cumlrkt::stream_of(handle);
  const filled<double> x(res, {0, 1, 2, 3}, {rows, 1});
  auto* data = static_cast<double*>(x.array.view.data);
  const qn_shape shape{rows, 1, 1, 2, true};
  const fit_options options{.penalty = CUML_LINEAR_MODEL_PENALTY_NONE,
                            .C = 1.0,
                            .l1_ratio = 0.0,
                            .tol = tol,
                            .max_iter = max_iter,
                            .linesearch_max_iter = linesearch_max_iter,
                            .lbfgs_memory = lbfgs_memory,
                            .fit_intercept = true,
                            .normalized = false,
                            .class_mode = 0};
  const column_stats stats{.sigma = {1.0},
                           .centred = {1.0},
                           .mean = {mean},
                           .magnitude = {largest},
                           .kept = {0},
                           .largest = largest};
  staged_fit<double> st{.copied = std::nullopt,
                        .source = data,
                        .y = to_device(std::vector<double>{0, 1, 0, 1}, stream),
                        .w = std::nullopt,
                        .kept = {0},
                        .shift = 0,
                        .pen = {0, 0},
                        .shape = shape,
                        .sigma = {1.0},
                        .mean = {mean}};
  const solve_check check{shape, {1.0}, {mean}, {0, 0}, first_iterations};
  const attempt<double> plain{
      {to_device(std::vector<double>(params, 0.0), stream), first_iterations},
      {first_worst, first_bound, 0.0},
      check};
  outcome out{false, 0, {}, {}};
  const qn_solver<double> solve =
      [&](const raft::handle_t&, double*, double*, double*, const qn_shape&,
          penalties, const fit_options& budget,
          const std::vector<double>& start) {
        out.starts.push_back(start);
        out.budgets.push_back(budget.max_iter);
        return qn_solution<double>{
            to_device(std::vector<double>(params, start.empty() ? nan : 0.0),
                      stream),
            cold_spends_the_budget ? budget.max_iter : 1};
      };
  const rescue<double> r =
      rescued<double>(handle, matrix<double, int64_t>{data, rows, 1},
                      row_selection{nullptr, rows}, stats, st, plain, options,
                      solve);
  out.rescued = r.solution.has_value();
  out.n_iter = r.n_iter;
  return out;
}

TEST(LogisticRescue, NonFiniteColdAttemptStillRunsTheWarmAttempt) {
  CUMLRKT_REQUIRE_GPU();
  const outcome out = rescued_with(false);
  EXPECT_FALSE(out.rescued);
  ASSERT_EQ(out.starts.size(), 2U);
  EXPECT_TRUE(out.starts[0].empty());
  EXPECT_EQ(out.starts[1].size(), static_cast<std::size_t>(params));
}

TEST(LogisticRescue, EachAttemptGetsTheIterationsLeft) {
  CUMLRKT_REQUIRE_GPU();
  const outcome out = rescued_with(false);
  EXPECT_EQ(out.budgets,
            (std::vector<int>{max_iter - first_iterations,
                              max_iter - first_iterations - 1}));
  EXPECT_EQ(out.n_iter, first_iterations + 2);
}

TEST(LogisticRescue, AColdAttemptThatSpendsTheBudgetSkipsTheWarmOne) {
  CUMLRKT_REQUIRE_GPU();
  const outcome out = rescued_with(true);
  EXPECT_FALSE(out.rescued);
  ASSERT_EQ(out.starts.size(), 1U);
  EXPECT_EQ(out.budgets, std::vector<int>{max_iter - first_iterations});
  EXPECT_EQ(out.n_iter, max_iter);
}

}  // namespace
