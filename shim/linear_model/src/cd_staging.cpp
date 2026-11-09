#include "cd_staging.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "raftrkt/view.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int32_t most_cd_rows = std::numeric_limits<int32_t>::max();

int half_floor(int x) {
  return even_floor(x) / 2;
}

long double value_of(scaled_sum s) {
  return s.value == 0
             ? 0.0L
             : std::ldexp(static_cast<long double>(s.value), s.exponent);
}

long double violation(long double cross, double coef, long double weight,
                      cd_penalty p) {
  const long double l1 = weight * p.l1;
  if (coef == 0) {
    return std::max(0.0L, std::abs(cross) - l1);
  }
  const long double l2 = weight * p.l2 * coef;
  return std::abs(cross - l2 -
                  std::copysign(l1, static_cast<long double>(coef)));
}

template <typename T>
void refuse_alpha(double alpha, int64_t rows, int shared) {
  throw logic_error(std::format(
      "alpha: expected at most {} for this X in {}, got {}",
      printed(std::ldexp(static_cast<double>(std::numeric_limits<T>::max()) /
                             static_cast<double>(rows),
                         -2 * shared)),
      raftrkt::dtype_name(raftrkt::dtype_code<T>), printed(alpha)));
}

}  // namespace

template <typename T>
int cd_shared_cap(const staging& s, int64_t rows) {
  constexpr int reach = std::numeric_limits<T>::max_exponent;
  const int beta = std::bit_width(static_cast<uint64_t>(rows));
  const int p = std::ranges::max(s.kept | std::views::transform([&](int64_t j) {
                                   return std::ilogb(s.largest[j]);
                                 }));
  const int q = -s.y_exponent;
  const int columns = std::ranges::min(
      s.kept | std::views::transform([&](int64_t j) {
        return unit_exponent(s.largest[j]) + column_reach<T>(rows);
      }));
  return std::min({columns, half_floor(reach - 6 - beta) - p,
                   half_floor(reach - 6 - beta - p - q), reach - 4 - beta - q});
}

template <typename T>
cd_scale<T> cd_scale_of(const staging& s, int64_t rows, double alpha) {
  const auto narrowest = std::ranges::max(
      s.kept, {}, [&](int64_t j) { return s.column_exponent[j]; });
  const int widest_exponent = s.column_exponent[narrowest];
  const int shared = std::min(widest_exponent, cd_shared_cap<T>(s, rows));
  if (shared < widest_exponent - narrowest_spread_bits) {
    throw logic_error(std::format(
        "X: column {} is too narrow beside the other columns and y for {} "
        "coordinate descent",
        narrowest, raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
  const auto staged = static_cast<T>(std::ldexp(alpha, 2 * shared));
  if (!std::isfinite(staged) || !std::isfinite(staged * static_cast<T>(rows))) {
    refuse_alpha<T>(alpha, rows, shared);
  }
  return {shared, staged};
}

kkt_terms kkt_of(const residual_terms& t, const std::vector<double>& coef,
                 double weight, cd_penalty penalty, cd_epochs epochs) {
  const long double y_square = value_of(t.y_square);
  const long double y_norm = std::sqrt(y_square);
  long double worst = 0;
  long double norms = 0;
  double largest = 0;
  for (size_t k = 0; k < t.moments.size(); ++k) {
    const long double square = value_of(t.moments[k].square);
    if (square == 0) {
      continue;
    }
    const long double v =
        violation(value_of(t.moments[k].cross), coef[k], weight, penalty);
    const long double norm = std::sqrt(square);
    const long double ratio =
        y_norm == 0 ? (v == 0 ? 0.0L : INFINITY) : v / (norm * y_norm);
    worst =
        std::isnan(ratio) || std::isnan(worst) ? NAN : std::max(worst, ratio);
    norms += norm;
    largest = std::max(largest, std::abs(coef[k]));
  }
  const long double spread =
      y_norm == 0 ? 0.0L : static_cast<long double>(largest) * norms / y_norm;
  const long double epochs_rows = static_cast<long double>(epochs.n_iter) *
                                  static_cast<long double>(epochs.rows);
  return {static_cast<double>(worst), static_cast<double>(epochs.tol * spread),
          static_cast<double>(epochs.roundoff * (1 + spread) *
                              std::sqrt(epochs_rows))};
}

void require_kkt(const kkt_terms& k, cd_epochs epochs) {
  const double bound = k.bound + k.allowance;
  if (k.residual > bound || std::isnan(k.residual)) {
    throw logic_error(std::format(
        "X: coordinate descent stopped after {} epochs with every coefficient "
        "under tol {} and KKT residual {} > {}; tol is absolute in the "
        "coefficients' units, so scale y up or X down",
        epochs.n_iter, printed(epochs.tol), printed(k.residual),
        printed(bound)));
  }
}

bool absolute_stop(int n_iter, int max_iter, double largest, double tol) {
  return n_iter < max_iter && largest > 0 && largest < tol;
}

void require_cd_rows(int64_t rows) {
  if (rows > most_cd_rows) {
    throw logic_error(std::format(
        "X: {} rows of positive weight exceed coordinate descent's {}", rows,
        most_cd_rows));
  }
}

void require_cd_codes(const cd_request& request) {
  const int32_t selection = request.fit.algorithm;
  if (selection != CUML_LINEAR_MODEL_CD_CYCLIC &&
      selection != CUML_LINEAR_MODEL_CD_RANDOM) {
    throw logic_error("selection: expected cyclic or random");
  }
  require_fit_intercept(request.fit.fit_intercept);
  if (!std::isfinite(request.alpha) || !(request.alpha >= 0)) {
    throw logic_error(
        std::format("alpha: expected a finite value at least 0, got {}",
                    printed(request.alpha)));
  }
  if (!(request.l1_ratio >= 0 && request.l1_ratio <= 1)) {
    throw logic_error(
        std::format("l1-ratio: expected a finite value from 0 to 1, got {}",
                    printed(request.l1_ratio)));
  }
  if (request.max_iter < 1) {
    throw logic_error(
        std::format("max-iter: expected at least 1, got {}", request.max_iter));
  }
  if (!std::isfinite(request.tol) || !(request.tol >= 0)) {
    throw logic_error(
        std::format("tol: expected a finite value of at least 0, got {}",
                    printed(request.tol)));
  }
}

template int cd_shared_cap<float>(const staging&, int64_t);
template int cd_shared_cap<double>(const staging&, int64_t);
template cd_scale<float> cd_scale_of<float>(const staging&, int64_t, double);
template cd_scale<double> cd_scale_of<double>(const staging&, int64_t, double);

}  // namespace cumlrkt::glm
