#include "cd_staging.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
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
constexpr int moment_sum_guard_bits = 6;
constexpr int label_sum_guard_bits = 4;
constexpr long double coefficient_rounding = 2;

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
constexpr int penalty_exponent = std::numeric_limits<T>::max_exponent - 1;

template <typename T>
void refuse_alpha(double alpha, int64_t rows, int shared) {
  throw logic_error(
      std::format("alpha: expected at most {} for this X in {}, got {}",
                  printed(std::ldexp(1.0 / static_cast<double>(rows),
                                     penalty_exponent<T> - 2 * shared)),
                  raftrkt::dtype_name(raftrkt::dtype_code<T>), printed(alpha)));
}

}  // namespace

template <typename T>
int cd_shared_cap(const staging& s, int64_t rows) {
  constexpr int reach = std::numeric_limits<T>::max_exponent;
  const int row_bits = std::bit_width(static_cast<uint64_t>(rows));
  const int column_magnitude =
      std::ranges::max(s.kept | std::views::transform([&](int64_t j) {
                         return std::ilogb(s.largest[j]);
                       }));
  const int y_magnitude = -s.y_exponent;
  const int moments = reach - moment_sum_guard_bits - row_bits;
  return std::min({shared_cap<T>(s, rows),
                   half_floor(moments) - column_magnitude,
                   half_floor(moments - column_magnitude - y_magnitude),
                   reach - label_sum_guard_bits - row_bits - y_magnitude});
}

template <typename T>
cd_scale<T> cd_scale_of(const staging& s, int64_t rows, double alpha) {
  const auto narrowest = std::ranges::max(
      s.kept, {}, [&](int64_t j) { return s.column_exponent[j]; });
  const int narrowest_exponent = s.column_exponent[narrowest];
  const int shared = std::min(narrowest_exponent, cd_shared_cap<T>(s, rows));
  if (shared < narrowest_exponent - narrowest_column_slack_bits) {
    throw logic_error(std::format(
        "X: column {} is too narrow beside the other columns and y for {} "
        "coordinate descent",
        narrowest, raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
  const auto staged = static_cast<T>(std::ldexp(alpha, 2 * shared));
  // Cap (a) keeps cdFit's column sums of squares under 2^(E-4), so with
  // alpha'm <= 2^(E-1) its denominator sum + alpha'(1-rho)m stays finite.
  if (std::ldexp(
          static_cast<long double>(staged) * static_cast<long double>(rows),
          -penalty_exponent<T>) > 1) {
    refuse_alpha<T>(alpha, rows, shared);
  }
  return {shared, staged};
}

kkt_terms kkt_of(const residual_terms& t, const std::vector<double>& coef,
                 double weight, cd_penalty penalty, cd_epochs epochs) {
  const long double total = weight;
  std::vector<long double> sigma;
  sigma.reserve(t.moments.size());
  for (const moment& m : t.moments) {
    sigma.push_back(std::sqrt(value_of(m.square) / total));
  }
  const long double sigmas = std::accumulate(sigma.begin(), sigma.end(), 0.0L);
  const long double largest =
      std::ranges::max(coef | std::views::transform([](double c) {
                         return static_cast<long double>(std::abs(c));
                       }));
  std::optional<long double> worst = 0.0L;
  long double reach = 0;
  for (size_t k = 0; k < sigma.size(); ++k) {
    if (sigma[k] == 0) {
      continue;
    }
    const long double curvature =
        std::max((sigma[k] * sigma[k]) + penalty.l2, sigma[k] * sigmas);
    const long double move =
        violation(value_of(t.moments[k].cross), coef[k], total, penalty) /
        (total * curvature * largest);
    worst = worst && !std::isnan(move) ? std::optional(std::max(*worst, move))
                                       : std::nullopt;
    reach = std::max(reach, sigma[k] / curvature);
  }
  const long double y_sigma = std::sqrt(value_of(t.y_square) / total);
  const long double gradient =
      std::sqrt(static_cast<long double>(epochs.rows)) * y_sigma * reach /
      largest;
  return {
      worst ? std::optional(static_cast<double>(*worst)) : std::nullopt,
      epochs.tol,
      static_cast<double>(epochs.roundoff *
                          std::sqrt(static_cast<long double>(epochs.n_iter)) *
                          (coefficient_rounding + gradient))};
}

void require_kkt(const kkt_terms& k, int n_iter, double tol) {
  const double bound = k.bound + k.allowance;
  if (!k.residual || *k.residual > bound) {
    throw logic_error(std::format(
        "X: coordinate descent stopped after {} epochs with every coefficient "
        "under tol {} and KKT residual {} > {}; tol is absolute in the "
        "coefficients' units, so scale y up or X down",
        n_iter, printed(tol),
        k.residual ? printed(*k.residual) : "nan", printed(bound)));
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
  if (std::isnan(request.l1_ratio) || request.l1_ratio < 0 ||
      request.l1_ratio > 1) {
    throw logic_error(
        std::format("l1-ratio: expected a finite value from 0 to 1, got {}",
                    printed(request.l1_ratio)));
  }
  if (request.max_iter < 1) {
    throw logic_error(std::format(
        "max-iter: expected an exact integer from 1 to {}, got {}",
        std::numeric_limits<int32_t>::max(), request.max_iter));
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
