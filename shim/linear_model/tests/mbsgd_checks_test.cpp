#include "mbsgd_checks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"

namespace {

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
    EXPECT_EQ(refusal_of([&] { cumlrkt::glm::mbsgd::require_sgd_options(o); }),
              r.message);
  }
}

TEST(MbsgdOptions, ValuesOutsideTheXDtypeAreRefused) {
  using cumlrkt::glm::mbsgd::require_sgd_values;
  constexpr double above_float = 1e150;
  constexpr double below_float = 1e-50;
  constexpr double past_float = 1e40;
  sgd_options o = cuml_defaults;
  EXPECT_EQ(refusal_of([&] { require_sgd_values<float>(o); }), "accepted");
  o.alpha = above_float;
  EXPECT_EQ(refusal_of([&] { require_sgd_values<double>(o); }), "accepted");
  EXPECT_EQ(refusal_of([&] { require_sgd_values<float>(o); }),
            "alpha: 1e+150 is not representable in float32");
  o = cuml_defaults;
  o.eta0 = below_float;
  EXPECT_EQ(refusal_of([&] { require_sgd_values<float>(o); }),
            "eta0: 1e-50 is not representable in float32");
  o = cuml_defaults;
  o.power_t = past_float;
  EXPECT_EQ(refusal_of([&] { require_sgd_values<float>(o); }),
            "power-t: 1e+40 is not representable in float32");
  o = cuml_defaults;
  o.tol = 0;
  o.power_t = 0;
  EXPECT_EQ(refusal_of([&] { require_sgd_values<float>(o); }), "accepted");
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
  EXPECT_DOUBLE_EQ(
      l1_allowance(with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                 .alpha = 10,
                                 .l1_ratio = 0.15,
                                 .fit_intercept = 1,
                                 .eta0 = 1e-3}),
                   3, uncapped, false),
      3.5 * 4 * 1e-3 * 100);
  EXPECT_DOUBLE_EQ(
      l1_allowance(with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                 .alpha = 10,
                                 .l1_ratio = 0.15,
                                 .fit_intercept = 0,
                                 .eta0 = 1e-3}),
                   3, uncapped, false),
      3.5 * 3 * 1e-3 * 100);
  EXPECT_DOUBLE_EQ(
      l1_allowance(
          with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_ELASTICNET,
                        .alpha = 2,
                        .l1_ratio = 0.25,
                        .fit_intercept = 1,
                        .eta0 = 0.5}),
          30, uncapped, false),
      3.5 * 31 * 0.5 * 0.25);
  EXPECT_EQ(l1_allowance(with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L2,
                                       .alpha = 10,
                                       .l1_ratio = 1,
                                       .fit_intercept = 1,
                                       .eta0 = 1}),
                         3, uncapped, false),
            0);
  EXPECT_EQ(
      l1_allowance(with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_NONE,
                                 .alpha = 10,
                                 .l1_ratio = 1,
                                 .fit_intercept = 1,
                                 .eta0 = 1}),
                   3, uncapped, false),
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
  EXPECT_DOUBLE_EQ(l1_allowance(o, 3, (a / 4) + 1e-4, false), a);
  EXPECT_DOUBLE_EQ(l1_allowance(o, 3, (a / 4) - 1e-4, false), a - 4e-4);
  const double start = 0.25;
  EXPECT_DOUBLE_EQ(l1_allowance(o, 3, start, false), 1.0);
  EXPECT_FALSE(
      gate_refusal(true, start, 1.25624, l1_allowance(o, 3, start, false), 1e-3)
          .has_value());
  EXPECT_EQ(
      gate_refusal(true, start, 1.25626, l1_allowance(o, 3, start, false), 1e-3)
          .value_or(""),
      "the objective rose from 0.25 to 1.25626; lower #:eta0 (now 0.001)");
}

TEST(MbsgdGate, TheRegressorL1AllowanceIsUncapped) {
  using cumlrkt::glm::mbsgd::l1_allowance;
  const sgd_options o = with_penalty({.penalty = CUML_LINEAR_MODEL_PENALTY_L1,
                                      .alpha = 10,
                                      .l1_ratio = 0.15,
                                      .fit_intercept = 1,
                                      .eta0 = 1e-3});
  const double a = 3.5 * 4 * 1e-3 * 100;
  EXPECT_DOUBLE_EQ(l1_allowance(o, 3, a / 8, true), a);
  EXPECT_DOUBLE_EQ(l1_allowance(o, 3, a / 8, false), a / 2);
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
