#include "logistic_resolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <vector>

#include "cumlrkt/common.hpp"
#include "cumlrkt/support.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"

namespace cumlrkt::glm::logistic {

namespace {

constexpr double tolerance_floor = 1e-4;
constexpr double residual_factor = 64;
constexpr double mixed_spread_ratio = 0x1p8;

const char* cutoff_advice(const solve_check& c, const fit_options& o) {
  if (c.n_iter >= o.max_iter) {
    return "raise max-iter";
  }
  const auto [narrowest, widest] = std::ranges::minmax(c.sigma);
  if (widest > mixed_spread_ratio * narrowest) {
    return "standardise the columns";
  }
  return c.shape.fit_intercept ? "lower tol, or centre X"
                               : "lower tol, centre X, or fit an intercept";
}

}  // namespace

resolution resolution_of(const residual_sums& sums,
                         const std::vector<double>& params,
                         const fit_options& options, const solve_check& c) {
  const auto m = static_cast<double>(c.shape.m);
  const auto per_row = [m](double v) { return v / m; };
  const penalties pen = options.normalized
                            ? penalties{per_row(c.pen.l1), per_row(c.pen.l2)}
                            : c.pen;
  const int64_t k = c.shape.k;
  double f = per_row(sums.loss);
  double worst = 0;
  for (int64_t q = 0; q < c.shape.p * k; ++q) {
    const double v = params[q];
    f += (pen.l1 * std::abs(v)) + (pen.l2 / 2 * v * v);
    const double centred = c.shape.fit_intercept
                               ? c.mean[q / k] * per_row(sums.intercept[q % k])
                               : 0.0;
    const double g = per_row(sums.gradient[q]) - centred + (pen.l2 * v);
    const double pg =
        v != 0 ? g + std::copysign(pen.l1, v)
               : std::copysign(std::max(std::abs(g) - pen.l1, 0.0), g);
    worst = std::max(worst, std::abs(pg) / c.sigma[q / k]);
  }
  for (int64_t q = 0; c.shape.fit_intercept && q < k; ++q) {
    worst = std::max(worst, std::abs(per_row(sums.intercept[q])));
  }
  const double tau = std::max(options.tol, tolerance_floor);
  return {worst, residual_factor * tau * std::max(f, tau), f};
}

void refuse_cutoff(const resolution& r, const fit_options& options,
                   const solve_check& c) {
  throw logic_error(std::format(
      "the solver stopped at first-order residual {} above the bound {} "
      "after {} iterations (max-iter {}, tol {}); {}",
      printed(r.worst), printed(r.bound), c.n_iter, options.max_iter,
      printed(options.tol), cutoff_advice(c, options)));
}

}  // namespace cumlrkt::glm::logistic
