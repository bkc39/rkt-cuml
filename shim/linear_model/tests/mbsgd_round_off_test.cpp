#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "mbsgd_checks.hpp"

namespace {

using cumlrkt::glm::mbsgd::baseline;
using cumlrkt::glm::mbsgd::l1_allowance;
using cumlrkt::glm::mbsgd::round_off_prediction_refusal;
using cumlrkt::glm::mbsgd::round_off_refusal;
using cumlrkt::glm::mbsgd::sgd_options;

constexpr double y_floor = 4e-13;
constexpr double c = 1.7;
constexpr baseline constant{
    .start = y_floor * 32, .floor = y_floor, .uncentred = c * c, .centre = c};

constexpr sgd_options lasso_defaults{
    .loss = CUML_LINEAR_MODEL_SGD_SQUARED,
    .penalty = CUML_LINEAR_MODEL_PENALTY_L1,
    .alpha = 0.5,
    .l1_ratio = 0.15,
    .fit_intercept = 1,
    .epochs = 1000,
    .tol = 1e-3,
    .shuffle = 1,
    .learning_rate = CUML_LINEAR_MODEL_SGD_CONSTANT,
    .eta0 = 0.2,
    .power_t = 0.5,
    .batch_size = 32,
    .n_iter_no_change = 5,
    .seeded = 0,
    .seed = 0};
constexpr double past_f64_round_off = 33;
constexpr double past_f32_round_off = 1.01;

sgd_options lasso(int32_t penalty) {
  sgd_options o = lasso_defaults;
  o.penalty = penalty;
  return o;
}

sgd_options without_intercept(sgd_options o) {
  o.fit_intercept = 0;
  return o;
}

TEST(MbsgdRoundOff, AConstantFitWithAnInterceptHasAnUncappedL1Allowance) {
  const sgd_options o = lasso(CUML_LINEAR_MODEL_PENALTY_L1);
  const double a = 3.5 * 5 * 0.2 * 0.25;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 4, constant, true), a);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 4, constant, false),
                   std::min(a, c * c / 4));
}

TEST(MbsgdRoundOff, ACoefficientPastTwoPenaltyStepsIsRefused) {
  for (const auto& [penalty, r] :
       {std::pair{CUML_LINEAR_MODEL_PENALTY_L1, 1.0},
        std::pair{CUML_LINEAR_MODEL_PENALTY_ELASTICNET, 0.15}}) {
    const sgd_options o = lasso(penalty);
    const double step = 0.2 * 0.5 * r;
    EXPECT_FALSE(
        round_off_refusal<double>(o, constant, c, 1.99 * step).has_value());
    EXPECT_EQ(
        round_off_refusal<double>(o, constant, c, 2.01 * step).value_or(""),
        std::format("y is constant to round-off and a coefficient of the fit "
                    "reaches {:.6g}; lower #:eta0 or #:alpha",
                    2.01 * step));
  }
}

TEST(MbsgdRoundOff, AnInterceptOffByMoreThanAMillionthOfTheConstantIsRefused) {
  const sgd_options o = lasso(CUML_LINEAR_MODEL_PENALTY_L1);
  EXPECT_FALSE(
      round_off_refusal<double>(o, constant, c + (0.99e-6 * c), 0).has_value());
  EXPECT_EQ(
      round_off_refusal<double>(o, constant, c - (1.01e-6 * c), 0).value_or(""),
      "y is constant to round-off and the fit's intercept is off by "
      "-1.717e-06; lower #:eta0 or #:alpha");
}

TEST(MbsgdRoundOff, TheBoundsJudgeOnlyAnL1FitOfAConstantWithAnIntercept) {
  const double far = 1;
  EXPECT_FALSE(round_off_refusal<double>(
                   without_intercept(lasso(CUML_LINEAR_MODEL_PENALTY_L1)),
                   constant, far, far)
                   .has_value());
  EXPECT_FALSE(round_off_refusal<double>(lasso(CUML_LINEAR_MODEL_PENALTY_L2),
                                         constant, far, far)
                   .has_value());
  baseline spread = constant;
  spread.start = y_floor * past_f64_round_off;
  EXPECT_FALSE(round_off_refusal<double>(lasso(CUML_LINEAR_MODEL_PENALTY_L1),
                                         spread, far, far)
                   .has_value());
  spread.start = y_floor * past_f32_round_off;
  EXPECT_FALSE(round_off_refusal<float>(lasso(CUML_LINEAR_MODEL_PENALTY_L1),
                                        spread, far, far)
                   .has_value());
  EXPECT_TRUE(round_off_refusal<double>(lasso(CUML_LINEAR_MODEL_PENALTY_L1),
                                        spread, far, far)
                  .has_value());
}

TEST(MbsgdRoundOff, PredictionsOffByMoreThanAShareOfTheConstantAreRefused) {
  const double edge = (c / 1024) + std::sqrt(y_floor);
  for (const int32_t penalty :
       {CUML_LINEAR_MODEL_PENALTY_L1, CUML_LINEAR_MODEL_PENALTY_ELASTICNET}) {
    const sgd_options o = lasso(penalty);
    EXPECT_FALSE(round_off_prediction_refusal<double>(o, constant,
                                                      std::pow(0.99 * edge, 2))
                     .has_value());
    EXPECT_EQ(
        round_off_prediction_refusal<double>(o, constant,
                                             std::pow(1.01 * edge, 2))
            .value_or(""),
        std::format("y is constant to round-off and the fit's predictions are "
                    "off by {:.6g}; lower #:eta0 or #:alpha",
                    1.01 * edge));
  }
  for (const int32_t penalty :
       {CUML_LINEAR_MODEL_PENALTY_L2, CUML_LINEAR_MODEL_PENALTY_NONE}) {
    SCOPED_TRACE(
        "the function alone: the objective bound keeps an l2 or unpenalised "
        "fit far below this loss");
    EXPECT_EQ(round_off_prediction_refusal<double>(lasso(penalty), constant,
                                                   std::pow(1.01 * edge, 2))
                  .value_or(""),
              std::format("y is constant to round-off and the fit's "
                          "predictions are off by {:.6g}; lower #:eta0",
                          1.01 * edge));
  }
}

TEST(MbsgdRoundOff, ThePredictionBoundJudgesOnlyAFitWithAnInterceptAtRoundOff) {
  const double far = 1;
  EXPECT_FALSE(
      round_off_prediction_refusal<double>(
          without_intercept(lasso(CUML_LINEAR_MODEL_PENALTY_L2)), constant, far)
          .has_value());
  baseline spread = constant;
  spread.start = y_floor * past_f64_round_off;
  EXPECT_FALSE(round_off_prediction_refusal<double>(
                   lasso(CUML_LINEAR_MODEL_PENALTY_L2), spread, far)
                   .has_value());
  EXPECT_TRUE(round_off_prediction_refusal<double>(
                  lasso(CUML_LINEAR_MODEL_PENALTY_L2), constant, far)
                  .has_value());
  const baseline zero{.start = 0, .floor = 0, .uncentred = 0, .centre = 0};
  EXPECT_FALSE(round_off_prediction_refusal<float>(
                   lasso(CUML_LINEAR_MODEL_PENALTY_L1), zero, 0)
                   .has_value());
  EXPECT_TRUE(round_off_prediction_refusal<float>(
                  lasso(CUML_LINEAR_MODEL_PENALTY_L1), zero, 1e-30)
                  .has_value());
}

}  // namespace
