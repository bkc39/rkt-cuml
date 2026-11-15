#pragma once

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"

namespace cumlrkt::glm::mbsgd {

inline constexpr int64_t most_elements = std::numeric_limits<int32_t>::max();

struct sgd_options {
  int32_t loss;
  int32_t penalty;
  double alpha;
  double l1_ratio;
  int32_t fit_intercept;
  int32_t epochs;
  double tol;
  int32_t shuffle;
  int32_t learning_rate;
  double eta0;
  double power_t;
  int32_t batch_size;
  int32_t n_iter_no_change;
  int32_t seeded;
  uint32_t seed;
};

inline void require_flag(const char* what, int32_t flag) {
  if ((flag & ~1) != 0) {
    throw logic_error(std::format("{}: expected #t or #f", what));
  }
}

inline void require_positive_finite(const char* what, double got) {
  if (!(got > 0) || !std::isfinite(got)) {
    throw logic_error(std::format("{}: expected a positive finite real, got {}",
                                  what, printed(got)));
  }
}

inline void require_nonnegative_finite(const char* what, double got) {
  if (!(got >= 0) || !std::isfinite(got)) {
    throw logic_error(std::format(
        "{}: expected a nonnegative finite real, got {}", what, printed(got)));
  }
}

inline void require_sgd_options(const sgd_options& o) {
  if (o.loss < CUML_LINEAR_MODEL_SGD_SQUARED ||
      o.loss > CUML_LINEAR_MODEL_SGD_HINGE) {
    throw logic_error("loss: expected one of 'hinge 'log 'squared-loss");
  }
  if (o.penalty < CUML_LINEAR_MODEL_PENALTY_NONE ||
      o.penalty > CUML_LINEAR_MODEL_PENALTY_ELASTICNET) {
    throw logic_error("penalty: expected one of 'none 'l1 'l2 'elasticnet");
  }
  require_positive_finite("alpha", o.alpha);
  if (!(o.l1_ratio >= 0 && o.l1_ratio <= 1)) {
    throw logic_error(std::format("l1-ratio: expected a real in [0, 1], got {}",
                                  printed(o.l1_ratio)));
  }
  require_flag("fit-intercept", o.fit_intercept);
  require_at_least("epochs", o.epochs, 1);
  require_nonnegative_finite("tol", o.tol);
  require_flag("shuffle", o.shuffle);
  if (o.learning_rate < CUML_LINEAR_MODEL_SGD_CONSTANT ||
      o.learning_rate > CUML_LINEAR_MODEL_SGD_ADAPTIVE) {
    throw logic_error(
        "learning-rate: expected one of 'constant 'invscaling 'adaptive");
  }
  require_positive_finite("eta0", o.eta0);
  require_nonnegative_finite("power-t", o.power_t);
  require_at_least("batch-size", o.batch_size, 1);
  require_at_least("n-iter-no-change", o.n_iter_no_change, 1);
  require_flag("seeded", o.seeded);
}

inline void require_sgd_shape(int64_t n, int64_t d, int64_t y_len) {
  require_some("X", "row", n, 2);
  require_some("X", "column", d);
  if (y_len != n) {
    throw logic_error(
        std::format("y: expected {} values (X's rows), got {}", n, y_len));
  }
}

inline void require_elements(int64_t n, int64_t cols) {
  if (n > most_elements / cols) {
    throw logic_error(std::format("X: expected at most {} elements, got {}",
                                  most_elements, n * cols));
  }
}

inline void require_two_classes(int64_t k) {
  if (k != 2) {
    throw logic_error(std::format("y: expected exactly 2 classes, got {}", k));
  }
}

inline constexpr double gate_slack = 0.025;
inline constexpr double gate_l1_factor = 3.5;
inline constexpr double gate_floor_share = 1e-12;
inline constexpr double gate_round_off_factor = 10;

inline double l1_share(const sgd_options& o) {
  switch (o.penalty) {
    case CUML_LINEAR_MODEL_PENALTY_L1:
      return 1;
    case CUML_LINEAR_MODEL_PENALTY_ELASTICNET:
      return o.l1_ratio;
    default:
      return 0;
  }
}

// At a zero optimum l1 SGD oscillates, each coordinate moving about
// eta0·alpha·r: the penalty rises by alpha·r·|w|_1 and the loss can fall by
// no more, a rise of about 2·k·eta0·(alpha·r)^2. eta0 bounds the step under
// every schedule.
inline double l1_allowance(const sgd_options& o, int64_t d) {
  const auto k = static_cast<double>(d + (o.fit_intercept != 0 ? 1 : 0));
  const double a = o.alpha * l1_share(o);
  return gate_l1_factor * k * o.eta0 * a * a;
}

inline double classifier_floor(double start) {
  return gate_floor_share * start;
}

template <typename T>
double regressor_floor(double start, double mean_square) {
  constexpr double eps = std::numeric_limits<T>::epsilon();
  return (gate_floor_share * start) +
         (gate_round_off_factor * eps * eps * mean_square);
}

inline std::optional<std::string> gate_refusal(bool finite, double start,
                                               double end, double allowance,
                                               double eta0) {
  if (!finite) {
    return std::format("the solution is not finite; lower #:eta0 (now {})",
                       printed(eta0));
  }
  if (!(end <= (start * (1 + gate_slack)) + allowance)) {
    return std::format(
        "the objective rose from {:.6g} to {:.6g}; lower #:eta0 (now {})",
        start, end, printed(eta0));
  }
  return std::nullopt;
}

}  // namespace cumlrkt::glm::mbsgd
