#include "ridge_staging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
int alpha_exponent(double alpha, int weight_power) {
  return even_floor(
             (2 * ridge_headroom<T>)-(std::ilogb(alpha) + weight_power)) /
         2;
}

int shared_spread(const staging& s) {
  return std::ranges::min(s.kept | std::views::transform([&](int64_t j) {
                            return s.column_exponent[j];
                          }));
}

template <typename T>
auto negligible_against(const staging& s, int shared, ridge_penalty penalty) {
  int alpha_power = 0;
  const double alpha_mantissa = std::frexp(penalty.alpha, &alpha_power);
  const int scale_power = alpha_power + (2 * shared) + penalty.weight_power;
  return [&s, shared, alpha_mantissa, scale_power](int64_t j) {
    if (s.spread[j] == 0) {
      return false;
    }
    int spread_power = 0;
    const double spread_mantissa = std::frexp(s.spread[j], &spread_power);
    const int staged_spread_power =
        spread_power + shared - unit_exponent(s.largest[j]);
    return std::ldexp(alpha_mantissa / spread_mantissa / spread_mantissa,
                      scale_power - (2 * staged_spread_power)) <=
           unit_roundoff<T>;
  };
}

template <typename T>
T staged_alpha(double alpha, int power) {
  const auto staged = static_cast<T>(std::ldexp(alpha, power));
  if (!std::isfinite(staged)) {
    throw logic_error(std::format(
        "alpha: expected at most {} for this X in {}, got {}",
        printed(std::ldexp(static_cast<double>(std::numeric_limits<T>::max()),
                           -power)),
        raftrkt::dtype_name(raftrkt::dtype_code<T>), printed(alpha)));
  }
  return staged;
}

}  // namespace

template <typename T>
std::optional<ridge_scale<T>> ridge_scale_of(const staging& s, int64_t n,
                                             ridge_penalty penalty) {
  const auto [alpha, weight_power] = penalty;
  if (alpha == 0 || s.kept.empty()) {
    return std::nullopt;
  }
  const int spread = shared_spread(s);
  const int shared = std::min({spread + ridge_headroom<T>, shared_cap<T>(s, n),
                               std::max(alpha_exponent<T>(alpha, weight_power),
                                        spread + ridge_floor<T>)});
  const T staged = staged_alpha<T>(alpha, (2 * shared) + weight_power);
  if (std::ranges::all_of(s.kept, negligible_against<T>(s, shared, penalty))) {
    return std::nullopt;
  }
  return ridge_scale<T>{shared, staged};
}

int ridge_algorithm(int solver, int64_t rows, int64_t kept) {
  if (kept == 1) {
    return CUML_LINEAR_MODEL_RIDGE_SVD;
  }
  if (solver == CUML_LINEAR_MODEL_RIDGE_SVD && rows < kept) {
    return CUML_LINEAR_MODEL_RIDGE_EIG;
  }
  return solver;
}

int lstsq_algorithm(int solver) {
  return solver == CUML_LINEAR_MODEL_RIDGE_SVD ? CUML_LINEAR_MODEL_OLS_SVD
                                               : CUML_LINEAR_MODEL_OLS_EIG;
}

void require_ridge_codes(const ridge_request& request) {
  const int32_t solver = request.fit.algorithm;
  if (solver != CUML_LINEAR_MODEL_RIDGE_SVD &&
      solver != CUML_LINEAR_MODEL_RIDGE_EIG) {
    throw logic_error("solver: expected auto, eig or svd");
  }
  require_fit_intercept(request.fit.fit_intercept);
  if (!std::isfinite(request.alpha) || !(request.alpha >= 0)) {
    throw logic_error(
        std::format("alpha: expected a finite value at least 0, got {}",
                    printed(request.alpha)));
  }
}

template std::optional<ridge_scale<float>> ridge_scale_of<float>(const staging&,
                                                                 int64_t,
                                                                 ridge_penalty);
template std::optional<ridge_scale<double>> ridge_scale_of<double>(
    const staging&, int64_t, ridge_penalty);

}  // namespace cumlrkt::glm
