#include "mbsgd_checks.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"

namespace {

using cumlrkt::glm::mbsgd::baseline;
using cumlrkt::glm::mbsgd::sgd_options;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();

constexpr sgd_options cuml_defaults{
    .loss = CUML_LINEAR_MODEL_SGD_HINGE,
    .penalty = CUML_LINEAR_MODEL_PENALTY_L2,
    .alpha = 1e-4,
    .l1_ratio = 0.15,
    .fit_intercept = 1,
    .epochs = 1000,
    .tol = 1e-3,
    .shuffle = 1,
    .learning_rate = CUML_LINEAR_MODEL_SGD_CONSTANT,
    .eta0 = 1e-3,
    .power_t = 0.5,
    .batch_size = 32,
    .n_iter_no_change = 5,
    .seeded = 0,
    .seed = 0};
constexpr double negative_eta0 = -0.01;

std::string refusal_of(const std::function<void()>& check) {
  try {
    check();
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(MbsgdOptions, EachCodeAndBoundIsRefusedBeforeTheSolver) {
  struct row {
    std::function<void(sgd_options&)> change;
    std::string message;
  };
  const std::vector<row> rows{
      {[](sgd_options&) {}, "accepted"},
      {[](sgd_options& o) { o.loss = 3; },
       "loss: expected one of 'hinge 'log 'squared-loss"},
      {[](sgd_options& o) { o.loss = -1; },
       "loss: expected one of 'hinge 'log 'squared-loss"},
      {[](sgd_options& o) { o.penalty = 4; },
       "penalty: expected one of 'none 'l1 'l2 'elasticnet"},
      {[](sgd_options& o) { o.alpha = 0; },
       "alpha: expected a positive finite real, got 0"},
      {[](sgd_options& o) { o.alpha = nan; },
       "alpha: expected a positive finite real, got nan"},
      {[](sgd_options& o) { o.l1_ratio = 2; },
       "l1-ratio: expected a real in [0, 1], got 2"},
      {[](sgd_options& o) { o.l1_ratio = nan; },
       "l1-ratio: expected a real in [0, 1], got nan"},
      {[](sgd_options& o) { o.fit_intercept = 2; },
       "fit-intercept: expected #t or #f"},
      {[](sgd_options& o) { o.epochs = 0; },
       "epochs: expected at least 1, got 0"},
      {[](sgd_options& o) { o.tol = -1; },
       "tol: expected a nonnegative finite real, got -1"},
      {[](sgd_options& o) { o.tol = inf; },
       "tol: expected a nonnegative finite real, got inf"},
      {[](sgd_options& o) { o.shuffle = -1; }, "shuffle: expected #t or #f"},
      {[](sgd_options& o) { o.learning_rate = 0; },
       "learning-rate: expected one of 'constant 'invscaling 'adaptive"},
      {[](sgd_options& o) { o.eta0 = negative_eta0; },
       "eta0: expected a positive finite real, got -0.01"},
      {[](sgd_options& o) { o.power_t = -1; },
       "power-t: expected a nonnegative finite real, got -1"},
      {[](sgd_options& o) { o.batch_size = -1; },
       "batch-size: expected at least 1, got -1"},
      {[](sgd_options& o) { o.n_iter_no_change = 0; },
       "n-iter-no-change: expected at least 1, got 0"},
      {[](sgd_options& o) { o.seeded = 3; }, "seeded: expected #t or #f"},
  };
  for (const row& r : rows) {
    sgd_options o = cuml_defaults;
    r.change(o);
    EXPECT_EQ(refusal_of(
                  [&] { cumlrkt::glm::mbsgd::require_sgd_options<double>(o); }),
              r.message);
  }
}

constexpr double past_float = 1e150;
constexpr double above_float = 1e40;
constexpr double rounds_to_float_max = 3.40282356e38;
constexpr double float_subnormal = 1e-40;
constexpr double under_float = 1e-50;
constexpr double double_subnormal = 1e-310;

TEST(MbsgdOptions, ValuesTheXDtypeCannotHoldAreRefused) {
  using cumlrkt::glm::mbsgd::require_sgd_options;
  struct row {
    std::function<void(sgd_options&)> change;
    std::string as_float;
    std::string as_double;
  };
  const std::vector<row> rows = {
      {[](sgd_options&) {}, "accepted", "accepted"},
      {[](sgd_options& o) { o.alpha = past_float; },
       "alpha: 1e+150 overflows float32", "accepted"},
      {[](sgd_options& o) { o.alpha = rounds_to_float_max; }, "accepted",
       "accepted"},
      {[](sgd_options& o) { o.alpha = under_float; },
       "alpha: 1e-50 rounds to 0 in float32", "accepted"},
      {[](sgd_options& o) { o.tol = above_float; },
       "tol: 1e+40 overflows float32", "accepted"},
      {[](sgd_options& o) { o.tol = float_subnormal; }, "accepted", "accepted"},
      {[](sgd_options& o) { o.tol = under_float; }, "accepted", "accepted"},
      {[](sgd_options& o) { o.tol = double_subnormal; }, "accepted",
       "accepted"},
      {[](sgd_options& o) { o.tol = 0; }, "accepted", "accepted"},
      {[](sgd_options& o) { o.eta0 = under_float; },
       "eta0: 1e-50 rounds to 0 in float32", "accepted"},
      {[](sgd_options& o) { o.eta0 = double_subnormal; },
       "eta0: 1e-310 rounds to 0 in float32", "accepted"},
      {[](sgd_options& o) { o.power_t = above_float; },
       "power-t: 1e+40 overflows float32", "accepted"},
      {[](sgd_options& o) { o.power_t = 0; }, "accepted", "accepted"},
  };
  for (const row& r : rows) {
    sgd_options o = cuml_defaults;
    r.change(o);
    EXPECT_EQ(refusal_of([&] { require_sgd_options<float>(o); }), r.as_float);
    EXPECT_EQ(refusal_of([&] { require_sgd_options<double>(o); }), r.as_double);
  }
}

TEST(MbsgdOptions, AnEarlierValueOutsideTheDtypeIsRefusedFirst) {
  sgd_options o = cuml_defaults;
  o.alpha = past_float;
  o.eta0 = 0;
  EXPECT_EQ(
      refusal_of([&] { cumlrkt::glm::mbsgd::require_sgd_options<float>(o); }),
      "alpha: 1e+150 overflows float32");
}

TEST(MbsgdShape, RowsColumnsAndYLengthAreRefusedInOrder) {
  struct row {
    int64_t n;
    int64_t d;
    int64_t y;
    std::string message;
  };
  const std::vector<row> rows{
      {2, 1, 2, "accepted"},
      {1, 0, 3, "X: expected at least 2 rows, got 1"},
      {5, 0, 3, "X: expected at least 1 column, got 0"},
      {5, 3, 4, "y: expected 5 values (X's rows), got 4"},
      {2, 2147483647, 2, "accepted"},
  };
  for (const row& r : rows) {
    EXPECT_EQ(refusal_of([&] {
                cumlrkt::glm::mbsgd::require_sgd_shape(r.n, r.d, r.y);
              }),
              r.message);
  }
}

TEST(MbsgdShape, ElementsAndClassesAreRefused) {
  EXPECT_EQ(
      refusal_of([] { cumlrkt::glm::mbsgd::require_elements(1073741823, 2); }),
      "accepted");
  EXPECT_EQ(
      refusal_of([] { cumlrkt::glm::mbsgd::require_elements(1073741824, 2); }),
      "X: expected at most 2147483647 elements, got 2147483648");
  EXPECT_EQ(
      refusal_of([] { cumlrkt::glm::mbsgd::require_elements(2147483647, 1); }),
      "accepted");
  EXPECT_EQ(refusal_of([] { cumlrkt::glm::mbsgd::require_two_classes(2); }),
            "accepted");
  EXPECT_EQ(refusal_of([] { cumlrkt::glm::mbsgd::require_two_classes(1); }),
            "y: expected exactly 2 classes, got 1");
  EXPECT_EQ(refusal_of([] { cumlrkt::glm::mbsgd::require_two_classes(3); }),
            "y: expected exactly 2 classes, got 3");
}

TEST(MbsgdGate, NonFiniteAndRisingFitsAreRefusedWithTheAdvice) {
  using cumlrkt::glm::mbsgd::gate_refusal;
  EXPECT_FALSE(gate_refusal(true, 1.0, 1.0, 0, 0.01).has_value());
  EXPECT_FALSE(gate_refusal(true, 1.0, 0.25, 0, 0.01).has_value());
  EXPECT_EQ(gate_refusal(false, 1.0, 0.25, 0, 10).value_or(""),
            "the solution is not finite; lower #:eta0 (now 10)");
  EXPECT_EQ(
      gate_refusal(true, 0.6931471805599453, 3.0201, 0, 0.05).value_or(""),
      "the objective rose from 0.693147 to 3.0201; lower #:eta0 (now 0.05)");
  EXPECT_EQ(gate_refusal(true, 1.0, nan, 0, 0.05).value_or(""),
            "the objective rose from 1 to nan; lower #:eta0 (now 0.05)");
  EXPECT_EQ(gate_refusal(true, 1.0, nan, 1, 0.05).value_or(""),
            "the objective rose from 1 to nan; lower #:eta0 (now 0.05)");
}

TEST(MbsgdGate, ARiseWithinTheSlackOrTheFloorIsAccepted) {
  using cumlrkt::glm::mbsgd::gate_refusal;
  EXPECT_FALSE(gate_refusal(true, 1.0, 1.0249, 0, 0.01).has_value());
  EXPECT_EQ(gate_refusal(true, 1.0, 1.0251, 0, 0.01).value_or(""),
            "the objective rose from 1 to 1.0251; lower #:eta0 (now 0.01)");
  EXPECT_FALSE(gate_refusal(true, 1.0, 1.5249, 0.5, 0.01).has_value());
  EXPECT_EQ(gate_refusal(true, 1.0, 1.5251, 0.5, 0.01).value_or(""),
            "the objective rose from 1 to 1.5251; lower #:eta0 (now 0.01)");
  EXPECT_FALSE(gate_refusal(true, 0, 1e-20, 1e-12, 0.01).has_value());
  EXPECT_EQ(gate_refusal(true, 0, 2e-12, 1e-12, 0.01).value_or(""),
            "the objective rose from 0 to 2e-12; lower #:eta0 (now 0.01)");
  EXPECT_EQ(gate_refusal(true, 0, 1e-20, 0, 0.01).value_or(""),
            "the objective rose from 0 to 1e-20; lower #:eta0 (now 0.01)");
}

struct penalty_case {
  int32_t penalty;
  double alpha;
  double l1_ratio;
  int32_t fit_intercept;
  double eta0;
};

baseline above(double start) {
  return {start, 0, start};
}

sgd_options with_penalty(const penalty_case& c) {
  sgd_options o = cuml_defaults;
  o.penalty = c.penalty;
  o.alpha = c.alpha;
  o.l1_ratio = c.l1_ratio;
  o.fit_intercept = c.fit_intercept;
  o.eta0 = c.eta0;
  return o;
}

TEST(MbsgdGate, TheL1AllowanceIsCKEta0AlphaRSquared) {
  using cumlrkt::glm::mbsgd::l1_allowance;
  const double uncapped = 1e6;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(
                       with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                     .alpha = 10,
                                     .l1_ratio = 0.15,
                                     .fit_intercept = 1,
                                     .eta0 = 1e-3}),
                       3, above(uncapped), false),
                   3.5 * 4 * 1e-3 * 100);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(
                       with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                     .alpha = 10,
                                     .l1_ratio = 0.15,
                                     .fit_intercept = 0,
                                     .eta0 = 1e-3}),
                       3, above(uncapped), false),
                   3.5 * 3 * 1e-3 * 100);
  EXPECT_DOUBLE_EQ(
      l1_allowance<double>(
          with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_ELASTICNET,
                        .alpha = 2,
                        .l1_ratio = 0.25,
                        .fit_intercept = 1,
                        .eta0 = 0.5}),
          30, above(uncapped), false),
      3.5 * 31 * 0.5 * 0.25);
  EXPECT_EQ(l1_allowance<double>(
                with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L2,
                              .alpha = 10,
                              .l1_ratio = 1,
                              .fit_intercept = 1,
                              .eta0 = 1}),
                3, above(uncapped), false),
            0);
  EXPECT_EQ(l1_allowance<double>(
                with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_NONE,
                              .alpha = 10,
                              .l1_ratio = 1,
                              .fit_intercept = 1,
                              .eta0 = 1}),
                3, above(uncapped), false),
            0);
}

TEST(MbsgdGate, TheL1AllowanceIsCappedAtFourStarts) {
  using cumlrkt::glm::mbsgd::gate_refusal;
  using cumlrkt::glm::mbsgd::l1_allowance;
  const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                      .alpha = 10,
                                      .l1_ratio = 0.15,
                                      .fit_intercept = 1,
                                      .eta0 = 1e-3});
  const double a = 3.5 * 4 * 1e-3 * 100;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above((a / 4) + 1e-4), false), a);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above((a / 4) - 1e-4), false),
                   a - 4e-4);
  const double start = 0.25;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above(start), false), 1.0);
  EXPECT_FALSE(gate_refusal(true, start, 1.25624,
                            l1_allowance<double>(o, 3, above(start), false),
                            1e-3)
                   .has_value());
  EXPECT_EQ(
      gate_refusal(true, start, 1.25626,
                   l1_allowance<double>(o, 3, above(start), false), 1e-3)
          .value_or(""),
      "the objective rose from 0.25 to 1.25626; lower #:eta0 (now 0.001)");
}

TEST(MbsgdGate, TheRegressorL1AllowanceIsCappedAtAQuarterOfTheStart) {
  using cumlrkt::glm::mbsgd::l1_allowance;
  const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                      .alpha = 10,
                                      .l1_ratio = 0.15,
                                      .fit_intercept = 1,
                                      .eta0 = 1e-3});
  const double a = 3.5 * 4 * 1e-3 * 100;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above(a * 8), true), a);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above(a / 8), true), a / 32);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 3, above(a / 8), false), a / 2);
}

TEST(MbsgdGate, AnAlphaWhoseAllowanceOverflowsStaysFinite) {
  using cumlrkt::glm::mbsgd::gate_refusal;
  using cumlrkt::glm::mbsgd::l1_allowance;
  for (const bool regressor : {true, false}) {
    const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                        .alpha = 1e200,
                                        .l1_ratio = 0.15,
                                        .fit_intercept = 1,
                                        .eta0 = 1e-3});
    const double allowance = l1_allowance<double>(o, 3, above(2.0), regressor);
    EXPECT_TRUE(std::isfinite(allowance));
    EXPECT_TRUE(gate_refusal(true, 2.0, 1e197, allowance, o.eta0).has_value());
  }
}

TEST(MbsgdGate, AnUncheckableStartOrFloorIsRefused) {
  using cumlrkt::glm::mbsgd::scale_refusal;
  EXPECT_FALSE(scale_refusal(1e300, 1e300).has_value());
  EXPECT_EQ(scale_refusal(inf, 1).value_or(""),
            "y is too large to check the fit; rescale y");
  EXPECT_EQ(scale_refusal(1, inf).value_or(""),
            "y is too large to check the fit; rescale y");
  EXPECT_EQ(scale_refusal(nan, 1).value_or(""),
            "y is too large to check the fit; rescale y");
}

TEST(MbsgdGate, SumsTheFloat32SolverWouldOverflowAreRefusedByCause) {
  using cumlrkt::glm::mbsgd::magnitudes;
  using cumlrkt::glm::mbsgd::solver_scale_refusal;
  constexpr double limit = std::numeric_limits<float>::max() / 4;
  const std::string rescale_x =
      "X is too large for the solver's sums; rescale X";
  const std::string rescale_y = "y is too large to check the fit; rescale y";
  const std::string rescale_xy =
      "X·(y − ȳ) is too large for the solver's sums; rescale X or y";
  const magnitudes unit{.largest = 1, .heaviest_column = 1};
  const magnitudes half{.largest = 0.5, .heaviest_column = 1};
  EXPECT_FALSE(solver_scale_refusal<float>(unit, limit, 0, false).has_value());
  EXPECT_EQ(
      solver_scale_refusal<float>(unit, limit * 1.001, 0, false).value_or(""),
      rescale_xy);
  EXPECT_FALSE(
      solver_scale_refusal<float>(half, limit, limit, true).has_value());
  EXPECT_EQ(solver_scale_refusal<float>(unit, limit, 0, true).value_or(""),
            rescale_xy);
  EXPECT_EQ(
      solver_scale_refusal<float>(unit, 0, limit * 1.001, true).value_or(""),
      rescale_y);
  EXPECT_FALSE(
      solver_scale_refusal<float>(unit, 0, limit * 1.001, false).has_value());
  EXPECT_EQ(solver_scale_refusal<float>(unit, inf, 0, false).value_or(""),
            rescale_xy);
  EXPECT_FALSE(
      solver_scale_refusal<double>(unit, limit * 1e3, limit * 1e3, true)
          .has_value());
  EXPECT_EQ(solver_scale_refusal<double>(unit, inf, 0, false).value_or(""),
            rescale_xy);
  const magnitudes heavy{.largest = 1, .heaviest_column = limit * 1.001};
  EXPECT_EQ(solver_scale_refusal<float>(heavy, 0, 0, true).value_or(""),
            rescale_x);
  EXPECT_EQ(solver_scale_refusal<float>(heavy, 0, 0, false).value_or(""),
            rescale_x);
  EXPECT_EQ(
      solver_scale_refusal<float>(heavy, 0, limit * 1.001, true).value_or(""),
      rescale_x);
  const magnitudes edge{.largest = 1, .heaviest_column = limit};
  EXPECT_FALSE(solver_scale_refusal<float>(edge, 0, 0, true).has_value());
}

TEST(MbsgdGate, TheRoundOffMultipleIs32FloorsInFloat64AndOneInFloat32) {
  using cumlrkt::glm::mbsgd::at_round_off;
  const double floor = 4e-13;
  EXPECT_TRUE(at_round_off<double>({floor * 32, floor, 1}));
  EXPECT_FALSE(at_round_off<double>({floor * 32.01, floor, 1}));
  EXPECT_TRUE(at_round_off<float>({floor, floor, 1}));
  EXPECT_FALSE(at_round_off<float>({floor * 1.01, floor, 1}));
}

TEST(
    MbsgdGate,
    ARoundOffStartWithoutAnInterceptCapsTheL1AllowanceAtAQuarterOfMeanYSquared) {
  using cumlrkt::glm::mbsgd::l1_allowance;
  const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                      .alpha = 100,
                                      .l1_ratio = 0.15,
                                      .fit_intercept = 0,
                                      .eta0 = 0.1});
  const double a = 3.5 * 4 * 0.1 * 1e4;
  const double floor = 4e-13;
  const double uncentred = 2.89;
  for (const bool regressor : {true, false}) {
    EXPECT_DOUBLE_EQ(
        l1_allowance<double>(o, 4, {floor * 32, floor, uncentred}, regressor),
        uncentred / 4);
    EXPECT_DOUBLE_EQ(
        l1_allowance<float>(o, 4, {floor, floor, uncentred}, regressor),
        uncentred / 4);
    EXPECT_DOUBLE_EQ(
        l1_allowance<double>(o, 4, {floor * 32, floor, a * 8}, regressor), a);
  }
  EXPECT_DOUBLE_EQ(
      l1_allowance<float>(o, 4, {floor * 2, floor, uncentred}, true),
      floor * 2 / 4);
}

TEST(MbsgdGate, TheRegressorCapGrowsWithTheCoefficientsAbove800) {
  using cumlrkt::glm::mbsgd::l1_allowance;
  const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                      .alpha = 1,
                                      .l1_ratio = 0.15,
                                      .fit_intercept = 1,
                                      .eta0 = 1e-3});
  const double start = 3.0;
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 799, above(start), true),
                   0.25 * start);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 4000, above(start), true),
                   0.25 * (4001.0 / 800) * start);
  EXPECT_DOUBLE_EQ(l1_allowance<double>(o, 4000, above(start), false),
                   4 * start);
}

TEST(MbsgdGate, ARegressorLossAboveTheZeroModelsIsRefused) {
  using cumlrkt::glm::mbsgd::loss_refusal;
  const baseline base{.start = 1.5, .floor = 1e-3, .uncentred = 4};
  EXPECT_FALSE(
      loss_refusal<double>(base, (1.5 * 1.025) + 0.99e-3, 1e-3).has_value());
  EXPECT_TRUE(
      loss_refusal<double>(base, (1.5 * 1.025) + 1.01e-3, 1e-3).has_value());
  const baseline near_floor{.start = 1.5, .floor = 1, .uncentred = 4};
  EXPECT_FALSE(loss_refusal<float>(near_floor, 1.574, 1e-3).has_value());
  EXPECT_EQ(loss_refusal<float>(near_floor, 1.576, 1e-3).value_or(""),
            "the loss rose from 1.5 to 1.576; lower #:eta0 (now 0.001)");
  EXPECT_EQ(loss_refusal<double>(base, 2.1, 1e-3).value_or(""),
            "the loss rose from 1.5 to 2.1; lower #:eta0 (now 0.001)");
  EXPECT_EQ(loss_refusal<double>(base, nan, 1e-3).value_or(""),
            "the loss rose from 1.5 to nan; lower #:eta0 (now 0.001)");
}

TEST(MbsgdGate, TheLossGuardDoesNotApplyAtRoundOff) {
  using cumlrkt::glm::mbsgd::loss_refusal;
  const baseline constant{.start = 3e-33, .floor = 4e-32, .uncentred = 0.09};
  EXPECT_FALSE(loss_refusal<double>(constant, 1e-20, 1e-3).has_value());
  EXPECT_FALSE(
      loss_refusal<float>({1e-13, 1e-13, 2.89}, 1e-9, 1e-3).has_value());
  EXPECT_TRUE(
      loss_refusal<float>({1.1e-13, 1e-13, 2.89}, 1e-9, 1e-3).has_value());
}

TEST(MbsgdGate, AnInfiniteObjectiveIsRefusedWhateverTheAllowance) {
  using cumlrkt::glm::mbsgd::gate_refusal;
  EXPECT_EQ(gate_refusal(true, 0, inf, inf, 1e-3).value_or(""),
            "the objective rose from 0 to inf; lower #:eta0 (now 0.001)");
}

TEST(MbsgdGate, TheRegressorFloorIsRoundOffOfY) {
  using cumlrkt::glm::mbsgd::classifier_floor;
  using cumlrkt::glm::mbsgd::regressor_floor;
  constexpr double e64 = std::numeric_limits<double>::epsilon();
  constexpr double e32 = std::numeric_limits<float>::epsilon();
  EXPECT_DOUBLE_EQ(regressor_floor<double>(2.0, 1e12),
                   (1e-12 * 2.0) + (10 * e64 * e64 * 1e12));
  EXPECT_DOUBLE_EQ(regressor_floor<float>(2.0, 1e12),
                   (1e-12 * 2.0) + (10 * e32 * e32 * 1e12));
  EXPECT_DOUBLE_EQ(regressor_floor<double>(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(classifier_floor(0.5), 0.5e-12);
}

}  // namespace
