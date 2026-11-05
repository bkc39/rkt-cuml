#include "ridge_staging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <utility>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"

namespace cumlrkt::glm {

namespace {

int floor_half(int x) {
  return (x - (((x % 2) + 2) % 2)) / 2;
}

template <typename T>
int alpha_exponent(double alpha, int weight_power) {
  if (alpha == 0) {
    return std::numeric_limits<int>::max();
  }
  return floor_half((2 * ridge_headroom<T>) -
                    (std::ilogb(alpha) + weight_power));
}

}  // namespace

template <typename T>
ridge_stage<T> ridge_staging(staging s, double alpha, int weight_power,
                             int64_t n) {
  if (s.kept.empty()) {
    return {std::move(s), static_cast<T>(std::ldexp(alpha, weight_power))};
  }
  const int spread = std::ranges::min(
      s.kept | std::views::transform([&](int64_t j) {
        return s.column_exponent[j] + ridge_headroom<T>;
      }));
  const int cap = std::ranges::min(
      s.kept | std::views::transform([&](int64_t j) {
        return unit_exponent(s.largest[j]) + column_reach<T>(n);
      }));
  const int shared =
      std::min({spread, cap, alpha_exponent<T>(alpha, weight_power)});
  std::ranges::fill(s.column_exponent, shared);
  return {std::move(s),
          static_cast<T>(std::ldexp(alpha, (2 * shared) + weight_power))};
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

int64_t ridge_workspace(int algorithm, ols_shape shape, int64_t lwork) {
  const auto [m, d] = shape;
  const int64_t square =
      algorithm == CUML_LINEAR_MODEL_RIDGE_SVD ? d * d : 2 * d * d;
  return (m * d) + square + (3 * d) + lwork;
}

void require_ridge_codes(int32_t solver, int32_t fit_intercept, double alpha) {
  if (solver != CUML_LINEAR_MODEL_RIDGE_SVD &&
      solver != CUML_LINEAR_MODEL_RIDGE_EIG) {
    throw logic_error("solver: expected auto, eig or svd");
  }
  if (fit_intercept != 0 && fit_intercept != 1) {
    throw logic_error("fit-intercept: expected #t or #f");
  }
  if (!std::isfinite(alpha) || !(alpha >= 0)) {
    throw logic_error(std::format(
        "alpha: expected a finite value at least 0, got {}", printed(alpha)));
  }
}

void require_ridge_workspace(int algorithm, ols_shape shape, int64_t lwork) {
  const int64_t size = ridge_workspace(algorithm, shape, lwork);
  if (size > most_index) {
    throw logic_error(std::format(
        "X: {} columns need a {}-element {} workspace, more than {} elements",
        shape.d, size, algorithm_name(algorithm), most_index));
  }
}

template <typename T>
void require_ridge_solution(double cosine, int algorithm) {
  if (!(cosine <= std::ldexp(1.0, -tolerance_bits<T>))) {
    throw logic_error(std::format(
        "X: the {} solve is not a ridge solution (gradient cosine {} > 2^-{}); "
        "X is ill-conditioned for this alpha",
        algorithm_name(algorithm), printed(cosine), tolerance_bits<T>));
  }
}

template ridge_stage<float> ridge_staging<float>(staging, double, int,
                                                 int64_t);
template ridge_stage<double> ridge_staging<double>(staging, double, int,
                                                   int64_t);
template void require_ridge_solution<float>(double, int);
template void require_ridge_solution<double>(double, int);

}  // namespace cumlrkt::glm
