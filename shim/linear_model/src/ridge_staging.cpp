#include "ridge_staging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
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
  if (alpha == 0) {
    return std::numeric_limits<int>::max();
  }
  return even_floor((2 * ridge_headroom<T>)-(std::ilogb(alpha) +
                                             weight_power)) /
         2;
}

int shared_spread(const staging& s) {
  return std::ranges::min(s.kept | std::views::transform([&](int64_t j) {
                            return s.column_exponent[j];
                          }));
}

template <typename T>
int shared_cap(const staging& s, int64_t n) {
  return std::ranges::min(s.kept | std::views::transform([&](int64_t j) {
                            return unit_exponent(s.largest[j]) +
                                   column_reach<T>(n);
                          }));
}

double widest_spread(const staging& s, int shared) {
  return std::ranges::max(s.kept | std::views::transform([&](int64_t j) {
                            return std::ldexp(
                                s.spread[j],
                                shared - unit_exponent(s.largest[j]));
                          }));
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
ridge_scale<T> ridge_scale_of(const staging& s, int64_t n,
                              ridge_penalty penalty) {
  const auto [alpha, weight_power] = penalty;
  if (s.kept.empty()) {
    return {0, T{0}, true};
  }
  const int spread = shared_spread(s);
  const int shared =
      std::min({spread + ridge_headroom<T>, shared_cap<T>(s, n),
                std::max(alpha_exponent<T>(alpha, weight_power),
                         spread + ridge_floor<T>)});
  const T staged = staged_alpha<T>(alpha, (2 * shared) + weight_power);
  const double widest = widest_spread(s, shared);
  return {shared, staged,
          static_cast<double>(staged) / widest / widest <=
              std::ldexp(1.0, -std::numeric_limits<T>::digits)};
}

staging shared_at(staging s, int shared) {
  std::ranges::fill(s.column_exponent, shared);
  return s;
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
  return solver == CUML_LINEAR_MODEL_RIDGE_SVD ? CUML_LINEAR_MODEL_OLS_SVD_QR
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

template ridge_scale<float> ridge_scale_of<float>(const staging&, int64_t,
                                                  ridge_penalty);
template ridge_scale<double> ridge_scale_of<double>(const staging&, int64_t,
                                                    ridge_penalty);

}  // namespace cumlrkt::glm
