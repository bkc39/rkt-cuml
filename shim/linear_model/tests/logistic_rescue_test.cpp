#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <raft/core/handle.hpp>
#include <string>
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
using cumlrkt::glm::logistic::refuse_cutoff;
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

std::string first_refusal(const attempt<double>& plain,
                          const fit_options& options) {
  try {
    refuse_cutoff(plain.r, options, plain.check);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(LogisticRescue, NonFiniteColdAttemptKeepsTheFirstRefusal) {
  CUMLRKT_REQUIRE_GPU();
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
  std::vector<std::vector<double>> starts;
  const qn_solver<double> solve =
      [&](const raft::handle_t&, double*, double*, double*, const qn_shape&,
          penalties, const fit_options&, const std::vector<double>& start) {
        starts.push_back(start);
        return qn_solution<double>{
            to_device(std::vector<double>(params, start.empty() ? nan : 0.0),
                      stream),
            1};
      };
  std::string outcome = "accepted";
  try {
    if (!rescued<double>(handle, matrix<double, int64_t>{data, rows, 1},
                         row_selection{nullptr, rows}, stats, st, plain,
                         options, solve)) {
      refuse_cutoff(plain.r, options, plain.check);
    }
  } catch (const cumlrkt::logic_error& e) {
    outcome = e.what();
  }
  EXPECT_EQ(outcome, first_refusal(plain, options));
  ASSERT_EQ(starts.size(), 2U);
  EXPECT_TRUE(starts[0].empty());
  EXPECT_EQ(starts[1].size(), static_cast<std::size_t>(params));
}

}  // namespace
