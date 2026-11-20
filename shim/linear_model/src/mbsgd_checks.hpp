#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "mbsgd_magnitudes.hpp"
#include "min_rows.hpp"

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

template <typename T>
void require_in_dtype(const char* what, double got, bool positive) {
  const T t = static_cast<T>(got);
  const std::string dtype = raftrkt::dtype_name(raftrkt::dtype_code<T>);
  if (!std::isfinite(t)) {
    throw logic_error(
        std::format("{}: {} overflows {}", what, printed(got), dtype));
  }
  if (positive && t == 0) {
    throw logic_error(
        std::format("{}: {} rounds to 0 in {}", what, printed(got), dtype));
  }
}

template <typename T>
void require_positive_finite(const char* what, double got) {
  if (!(got > 0) || !std::isfinite(got)) {
    throw logic_error(std::format("{}: expected a positive finite real, got {}",
                                  what, printed(got)));
  }
  require_in_dtype<T>(what, got, true);
}

template <typename T>
void require_nonnegative_finite(const char* what, double got) {
  if (!(got >= 0) || !std::isfinite(got)) {
    throw logic_error(std::format(
        "{}: expected a nonnegative finite real, got {}", what, printed(got)));
  }
  require_in_dtype<T>(what, got, false);
}

template <typename T>
void require_sgd_options(const sgd_options& o) {
  if (o.loss < CUML_LINEAR_MODEL_SGD_SQUARED ||
      o.loss > CUML_LINEAR_MODEL_SGD_HINGE) {
    throw logic_error("loss: expected one of 'hinge 'log 'squared-loss");
  }
  if (o.penalty < CUML_LINEAR_MODEL_PENALTY_NONE ||
      o.penalty > CUML_LINEAR_MODEL_PENALTY_ELASTICNET) {
    throw logic_error("penalty: expected one of 'none 'l1 'l2 'elasticnet");
  }
  require_positive_finite<T>("alpha", o.alpha);
  if (!(o.l1_ratio >= 0 && o.l1_ratio <= 1)) {
    throw logic_error(std::format("l1-ratio: expected a real in [0, 1], got {}",
                                  printed(o.l1_ratio)));
  }
  require_flag("fit-intercept", o.fit_intercept);
  require_at_least("epochs", o.epochs, 1);
  require_nonnegative_finite<T>("tol", o.tol);
  require_flag("shuffle", o.shuffle);
  if (o.learning_rate < CUML_LINEAR_MODEL_SGD_CONSTANT ||
      o.learning_rate > CUML_LINEAR_MODEL_SGD_ADAPTIVE) {
    throw logic_error(
        "learning-rate: expected one of 'constant 'invscaling 'adaptive");
  }
  require_positive_finite<T>("eta0", o.eta0);
  require_nonnegative_finite<T>("power-t", o.power_t);
  require_at_least("batch-size", o.batch_size, 1);
  require_at_least("n-iter-no-change", o.n_iter_no_change, 1);
  require_flag("seeded", o.seeded);
}

inline void require_sgd_shape(int64_t n, int64_t d, int64_t y_len) {
  require_some("X", "row", n, min_rows);
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

// The loss guard refuses a regressor whose loss ends above start·(1 + slack)
// plus the floor, at most slack·start: worse than the zero model, with an
// intercept R² below −.025 − min(floor/start, .025), at worst −.05. Storing
// the fit's intercept b in the dtype moves it by at most ε/2·|b| ≈ ε/2·|ȳ|,
// which raises the loss by at most (ε/2·ȳ)², a fortieth of the floor's
// round-off term 10·ε²·mean(y²): at most floor/40 < start/40 = slack·start
// wherever the guard applies, above round-off (start > floor). The floor's
// share admits the float32 y = 1.7 + 1e-6·e elastic net, 2.6 floors above
// round-off, whose loss ends at 1.0251·start (F10-R round-off).
inline constexpr double gate_slack = 0.025;
inline constexpr double gate_l1_factor = 3.5;
inline constexpr double gate_l1_cap = 4;
inline constexpr double gate_l1_regressor_cap = 0.25;
// The oscillation's penalty grows with k while F_start does not, so above
// gate_l1_regressor_columns coefficients the regressor's cap grows with k:
// wide lassos whose predictions match the optimum's needed a cap of up to
// 0.25·k/875 (F10-W), and no fit with k at most 800, where every F10-R
// refusal lies, moves. The loss guard keeps a widened cap from passing a fit
// that predicts worse than the zero model.
inline constexpr double gate_l1_regressor_columns = 800;
inline constexpr double gate_floor_share = 1e-12;
inline constexpr double gate_round_off_factor = 10;
// A y equal to a constant c up to ±1 ulp has F_start ≤ ((1 + e)·ε·c)², e the
// error of the mean in units of ε·|c|, and the floor's round-off term is
// gate_round_off_factor·ε²·c². float32: the mean is summed in double, so
// e ≤ n·2^-53/ε ≤ 2 for n ≤ 2^31 even summed in sequence, and 3² < 10: a
// multiple of 1, which admits a spread of at most √10·ε·|ȳ|. float64:
// e ≤ log2(n)/2 ≤ 16 for a tree sum, and 17² < 32·10. An F_start at or below
// that many floors is round-off: the zero model fits y. Without fit-intercept
// A is capped at a share of the uncentred start, the zero model's loss
// (mean(y²) for the regressor), instead of a multiple of F_start. With it the
// zero model is b = ȳ, w = 0, and a fit is judged against it by the bounds
// below: an l1 or elastic-net fit by all three instead of by A, any other by
// A and the prediction bound. Under none, l2 or an l1-ratio of 0, A already
// holds √loss to 5.8·√floor (float64) or 1.42·√floor (float32), far below
// 2^-10·|ȳ|: A implies the prediction bound there, and in practice only l1
// and elastic-net fits meet its message.
template <typename T>
inline constexpr double gate_round_off_start =
    std::is_same_v<T, float> ? 1 : 32;
// The Racket tests restate gate_floor_share, gate_round_off_factor and
// gate_round_off_start above and the three round_off_ shares and steps below
// as literals; scripts/gate-constants.sh fails when they differ.
// A coordinate whose optimum is 0 moves by the penalty's step eta·α·r, at
// most eta0·α·r under every schedule, and by the loss gradient, which at
// round-off is y's rounding times x: it crosses 0 by at most one step, and
// one more covers the round-off (worst .95 steps, F10-R round-off).
inline constexpr double round_off_coef_steps = 2;
// cuML returns b = ȳ_T − x̄·w, ȳ_T the mean of y in the dtype. A float32
// constant's rounding is ε/2 ≈ 6e-8 of |ȳ|, and a tree-summed mean adds at
// most log2(n)/2·ε: within 8.4·ε for n up to 2^15.8 (worst 1.4e-7, F10-R
// round-off). x̄·w shrinks with eta0 and alpha, as the refusal advises.
inline constexpr double round_off_intercept_share = 1e-6;
// At round-off with fit-intercept the zero model b = ȳ, w = 0 predicts y to
// within √floor and R² measures rounding, not the fit: a fit under any
// penalty is refused when its prediction error √loss exceeds
// round_off_prediction_share·|ȳ| + √floor. The share lies between the
// largest error of a fit of a constant at the defaults with |ȳ| ≥ 0.3,
// 6.2e-7·|ȳ| (float32 l1 y = 0.3), and the smallest of a fit that must be
// refused, 0.040·|ȳ| (y = 1.7, l1 α .51 eta0 .1, r2; the intercept bound
// refuses it first). The smallest pinned row only this bound refuses is
// zero-mean elastic net α 1 eta0 .1, at 9.9e-3·|ȳ|. 2^-10 is 1,570 times the
// first and a 41st of the second, 2.6 octaves above their geometric middle
// 2^-12.6; it passes a near-constant fit within a few ulps whatever its R²
// (F10-R round-off). The oscillation of a constant's l1 fit has an absolute
// size, about eta0·α·r·√d, so a constant with |ȳ| below about
// 2^10·eta0·α·r·√d, 2e-4 at the defaults on standardised X with d = 4, is
// refused unless the fit is exact.
inline constexpr double round_off_prediction_share = 0.0009765625;
// sgdFit sums in X's dtype: X's columns and y for the centring means, and the
// products x·(y - ȳ) of a batch's gradient, X centred (|x - x̄| ≤ 2·max|x|).
// Each bound must stay solver_headroom below the dtype's largest value. The
// column bound holds without fit-intercept too: such a column has
// max|x| ≥ max/(4n), and a stable step, below 1/max|x|², rounds to 0 in the
// dtype for any n ≤ 2^31. The product bound sums the whole of y, so with
// batch-size below n it is stricter than the solver needs.
inline constexpr double solver_headroom = 4;

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

struct baseline {
  double start;
  double floor;
  double uncentred;
  double centre = 0;
};

template <typename T>
bool at_round_off(const baseline& base) {
  return base.start <= gate_round_off_start<T> * base.floor;
}

template <typename T>
bool fits_a_constant(const sgd_options& o, const baseline& base,
                     bool regressor) {
  return regressor && o.fit_intercept != 0 && l1_share(o) > 0 &&
         at_round_off<T>(base);
}

// At a zero optimum l1 SGD oscillates, each coordinate moving about
// eta0·alpha·r: the penalty rises by alpha·r·|w|_1 and the loss can fall by
// no more, a rise of about 2·k·eta0·(alpha·r)^2. eta0 bounds the step under
// every schedule. It is capped at a multiple of start, or of the uncentred
// start at round-off, so a large eta0 or alpha cannot hide divergence. A fit
// of a constant y is bounded by round_off_refusal and
// round_off_prediction_refusal instead.
template <typename T>
double l1_allowance(const sgd_options& o, int64_t d, const baseline& base,
                    bool regressor) {
  const auto k = static_cast<double>(d + (o.fit_intercept != 0 ? 1 : 0));
  const double a = o.alpha * l1_share(o);
  const double allowance = gate_l1_factor * k * o.eta0 * a * a;
  if (fits_a_constant<T>(o, base, regressor)) {
    return allowance;
  }
  if (at_round_off<T>(base)) {
    return std::min(allowance, gate_l1_regressor_cap * base.uncentred);
  }
  const double cap =
      regressor
          ? gate_l1_regressor_cap * std::max(1.0, k / gate_l1_regressor_columns)
          : gate_l1_cap;
  return std::min(allowance, cap * base.start);
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

inline std::optional<std::string> scale_refusal(double start, double floor) {
  if (!std::isfinite(start) || !std::isfinite(floor)) {
    return "y is too large to check the fit; rescale y";
  }
  return std::nullopt;
}

template <typename T>
std::optional<std::string> solver_scale_refusal(const magnitudes& x,
                                                double residual_mass,
                                                double y_mass, bool centred) {
  const double limit = std::numeric_limits<T>::max() / solver_headroom;
  if (!(x.heaviest_column <= limit)) {
    return "X is too large for the solver's sums; rescale X";
  }
  if (centred && !(y_mass <= limit)) {
    return "y is too large to check the fit; rescale y";
  }
  const double reach = centred ? 2 * x.largest : x.largest;
  if (!(reach * residual_mass <= limit)) {
    return "X·(y − ȳ) is too large for the solver's sums; rescale X or y";
  }
  return std::nullopt;
}

inline std::optional<std::string> gate_refusal(bool finite, double start,
                                               double end, double allowance,
                                               double eta0) {
  if (!finite) {
    return std::format("the solution is not finite; lower #:eta0 (now {})",
                       printed(eta0));
  }
  if (!std::isfinite(end) || !(end <= (start * (1 + gate_slack)) + allowance)) {
    return std::format(
        "the objective rose from {:.6g} to {:.6g}; lower #:eta0 (now {})",
        start, end, printed(eta0));
  }
  return std::nullopt;
}

template <typename T>
std::optional<std::string> round_off_refusal(const sgd_options& o,
                                             const baseline& base,
                                             double intercept,
                                             double largest_coef) {
  if (!fits_a_constant<T>(o, base, true)) {
    return std::nullopt;
  }
  const double step = o.eta0 * o.alpha * l1_share(o);
  if (!(largest_coef <= round_off_coef_steps * step)) {
    return std::format(
        "y is constant to round-off and a coefficient of the fit reaches "
        "{:.6g}; lower #:eta0 or #:alpha",
        largest_coef);
  }
  const double c = base.centre;
  if (!(std::abs(intercept - c) <= round_off_intercept_share * std::abs(c))) {
    return std::format(
        "y is constant to round-off and the fit's intercept is off by {:.6g}; "
        "lower #:eta0 or #:alpha",
        intercept - c);
  }
  return std::nullopt;
}

template <typename T>
std::optional<std::string> round_off_prediction_refusal(const sgd_options& o,
                                                        const baseline& base,
                                                        double loss) {
  if (o.fit_intercept == 0 || !at_round_off<T>(base)) {
    return std::nullopt;
  }
  const double off = std::sqrt(loss);
  if (off <= (round_off_prediction_share * std::abs(base.centre)) +
                 std::sqrt(base.floor)) {
    return std::nullopt;
  }
  return std::format(
      "y is constant to round-off and the fit's predictions are off by "
      "{:.6g}; {}",
      off, l1_share(o) > 0 ? "lower #:eta0 or #:alpha" : "lower #:eta0");
}

template <typename T>
std::optional<std::string> loss_refusal(const baseline& base, double loss,
                                        double eta0) {
  if (at_round_off<T>(base) ||
      loss <= (base.start * (1 + gate_slack)) +
                  std::min(base.floor, gate_slack * base.start)) {
    return std::nullopt;
  }
  return std::format(
      "the loss rose from {:.6g} to {:.6g}; lower #:eta0 (now {})", base.start,
      loss, printed(eta0));
}

}  // namespace cumlrkt::glm::mbsgd
