#include "glm_staging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <optional>
#include <raft/core/handle.hpp>
#include <ranges>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "raftrkt/view.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

namespace {

struct device_columns {
  buffer<int64_t> kept;
  buffer<int> exponent;

  staged_columns view() const {
    return {kept.data(), exponent.data(), static_cast<int64_t>(kept.size())};
  }
};

struct stationarity_terms {
  std::string_view solution;
  std::string_view residual;
  std::string_view cause;
};

device_columns columns_of(const staging& s, rmm::cuda_stream_view stream) {
  std::vector<int> e(s.kept.size());
  std::ranges::transform(s.kept, e.begin(),
                         [&](int64_t j) { return s.column_exponent[j]; });
  return {to_device(s.kept, stream), to_device(e, stream)};
}

template <typename T>
double largest_magnitude(const std::vector<T>& b) {
  double largest = 0;
  for (const T bk : b) {
    largest = std::max(largest, std::abs(static_cast<double>(bk)));
  }
  return largest;
}

}  // namespace

int even_floor(int x) {
  return x - (((x % 2) + 2) % 2);
}

double cosine_of(const moment& m, scaled_sum y) {
  const double ratio =
      (m.cross.value * m.cross.value) / (m.square.value * y.value);
  return std::sqrt(std::ldexp(
      ratio, (2 * m.cross.exponent) - m.square.exponent - y.exponent));
}

double stationarity_of(const moment& m, scaled_sum y, ridge_penalty p) {
  if (p.alpha == 0) {
    return cosine_of(m, y);
  }
  const double penalty = p.alpha * p.coefficient;
  const double widest = p.alpha * p.largest;
  if (!std::isfinite(widest)) {
    return INFINITY;
  }
  const int product = m.square.exponent + y.exponent;
  const int half = even_floor(product) / 2;
  const double norms =
      std::sqrt(std::ldexp(m.square.value * y.value, product - (2 * half)));
  const int top = widest == 0
                      ? std::max(m.cross.exponent, half)
                      : std::max({m.cross.exponent, half, std::ilogb(widest)});
  const double gradient = std::ldexp(m.cross.value, m.cross.exponent - top) -
                          std::ldexp(penalty, -top);
  const double size = std::ldexp(norms, half - top) + std::ldexp(widest, -top);
  return std::abs(gradient) / size;
}

std::string algorithm_name(int algorithm) {
  switch (algorithm) {
    case CUML_LINEAR_MODEL_OLS_SVD:
      return "svd";
    case CUML_LINEAR_MODEL_OLS_EIG:
      return "eig";
    case CUML_LINEAR_MODEL_OLS_QR:
      return "qr";
    default:
      return "svd-qr";
  }
}

std::string solver_name(glm_solver solver) {
  return solver.family == glm_family::cd ? "cd"
                                         : algorithm_name(solver.algorithm);
}

staging shared_at(staging s, int shared, std::optional<int> y_exponent) {
  std::ranges::fill(s.column_exponent, shared);
  s.y_exponent = y_exponent.value_or(s.y_exponent);
  return s;
}

int64_t ols_workspace(int algorithm, ols_shape shape, int64_t cusolver_lwork) {
  const auto [n, d] = shape;
  const int64_t m = std::min(n, d);
  switch (algorithm) {
    case CUML_LINEAR_MODEL_OLS_EIG:
      return (3 * d * d) + (2 * d);
    case CUML_LINEAR_MODEL_OLS_SVD:
      return cusolver_lwork + (n * m) + (d * m) + (2 * m) + 1;
    case CUML_LINEAR_MODEL_OLS_SVD_QR:
      return cusolver_lwork + (n * m) + (d * d) + (2 * m) + 1;
    default:
      return cusolver_lwork;
  }
}

template <typename T>
staging plan_staging(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const T* y, const T* w, row_selection rows,
                     bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const profiles profile = column_profiles(x, w, rows, fit_intercept, stream);
  std::vector<int> unit(x.d);
  std::ranges::transform(profile.columns, unit.begin(), [](column_profile p) {
    return unit_exponent(p.largest);
  });
  const std::vector<spread_terms> spread =
      column_spread(x, w, rows, unit, profile, stream);
  staging s;
  s.column_exponent.resize(x.d);
  s.largest.resize(x.d);
  s.spread.resize(x.d);
  s.normalised_mean.resize(x.d);
  s.normalised_remainder.resize(x.d);
  for (int64_t j = 0; j < x.d; ++j) {
    const column_profile& p = profile.columns[j];
    s.column_exponent[j] =
        spread_exponent<T>(unit[j], spread[j].spread, rows.count);
    s.largest[j] = p.largest;
    s.spread[j] = spread[j].spread;
    s.normalised_mean[j] = p.mean;
    s.normalised_remainder[j] = spread[j].remainder;
    if (p.varies) {
      s.kept.push_back(j);
    }
  }
  s.y_exponent = unit_exponent(largest_counted(y, x.n, w, stream));
  return s;
}

staging uniform(staging s) {
  if (!s.kept.empty()) {
    const int shared =
        std::ranges::min(s.kept | std::views::transform([&](int64_t j) {
                           return s.column_exponent[j];
                         }));
    std::ranges::fill(s.column_exponent, shared);
  }
  return s;
}

template <typename T>
buffer<T> staged_weights(const raft::handle_t& handle, const T* w, int64_t n,
                         double largest) {
  buffer<T> staged = scratch<T>(w == nullptr ? 0 : n, stream_of(handle));
  if (w != nullptr) {
    stage_elements(handle, w, w, row_selection{nullptr, n},
                   even_floor(unit_exponent(largest)), staged.data());
  }
  return staged;
}

template <typename T>
staged_problem<T> stage_problem(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, const staging& s,
                                row_selection rows) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  staged_problem<T> p{
      scratch<T>(rows.count * static_cast<int64_t>(s.kept.size()), stream),
      scratch<T>(rows.count, stream),
      scratch<T>(w == nullptr ? 0 : rows.count, stream)};
  const device_columns columns = columns_of(s, stream);
  stage_matrix(handle, x, w, rows, columns.view(), p.x.data());
  stage_elements(handle, y, w, rows, s.y_exponent, p.y.data());
  if (w != nullptr) {
    stage_elements(handle, w, w, rows, 0, p.w.data());
  }
  settle(handle);
  return p;
}

template <typename T>
int64_t unstage_coef(const raft::handle_t& handle, const T* staged,
                     const staging& s, int64_t d, T* coef) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  std::vector<int64_t> slot(d, -1);
  std::vector<int> shift(s.kept.size());
  for (size_t k = 0; k < s.kept.size(); ++k) {
    slot[s.kept[k]] = static_cast<int64_t>(k);
    shift[k] = s.column_exponent[s.kept[k]] - s.y_exponent;
  }
  const buffer<int64_t> slots = to_device(slot, stream);
  const buffer<int> shifts = to_device(shift, stream);
  unstage(handle, staged, slots.data(), shifts.data(), d, coef);
  const int64_t j = first_non_finite(coef, d, stream);
  settle(handle);
  return j;
}

template <typename T>
void require_finite_solution(const T* staged, int64_t d, T intercept,
                             glm_solver solver, rmm::cuda_stream_view stream) {
  if (first_non_finite(staged, d, stream) >= 0 || !std::isfinite(intercept)) {
    throw logic_error(
        std::format("X: the {} solve gave a non-finite coefficient",
                    solver_name(solver)));
  }
}

template <typename T>
void require_stationary(double cosine, glm_solver solver) {
  if (cosine <= std::ldexp(1.0, -tolerance_bits<T>)) {
    return;
  }
  const auto [solution, residual, cause] =
      solver.family == glm_family::ridge
          ? stationarity_terms{"ridge", "gradient",
                               "X is ill-conditioned for this alpha"}
          : stationarity_terms{"least-squares", "residual",
                               "X may have collinear columns"};
  throw logic_error(std::format(
      "X: the {} solve is not a {} solution ({} cosine {} > 2^-{}); {}",
      algorithm_name(solver.algorithm), solution, residual, printed(cosine),
      tolerance_bits<T>, cause));
}

template <typename T>
void require_in_range(int64_t column, T intercept) {
  if (column >= 0) {
    throw logic_error(
        std::format("X: column {}: the coefficient exceeds the {} range",
                    column, raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
  if (!std::isfinite(intercept)) {
    throw logic_error(std::format("y: the intercept exceeds the {} range",
                                  raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
}

template <typename T>
double stationarity_cosine(const raft::handle_t& handle,
                           const matrix<T, int64_t>& x, const T* y, const T* w,
                           row_selection rows, const staging& s, const T* coef,
                           bool fit_intercept, T alpha) {
  const residual_terms terms =
      residual_moments(handle, x, y, w, rows, s, coef, fit_intercept);
  const auto count = static_cast<int64_t>(terms.moments.size());
  const std::optional<std::vector<T>> b =
      alpha == 0 ? std::nullopt
                 : std::optional(to_host(coef, count, stream_of(handle)));
  const double largest = b ? largest_magnitude(*b) : 0.0;
  double worst = 0;
  for (int64_t k = 0; k < count; ++k) {
    if (terms.largest[k] == 0) {
      continue;
    }
    const double g =
        stationarity_of(terms.moments[k], terms.y_square,
                        {static_cast<double>(alpha),
                         b ? static_cast<double>((*b)[k]) : 0.0, largest});
    worst = std::isnan(g) || std::isnan(worst) ? NAN : std::max(worst, g);
  }
  return worst;
}

template buffer<float> staged_weights<float>(const raft::handle_t&,
                                             const float*, int64_t, double);
template buffer<double> staged_weights<double>(const raft::handle_t&,
                                               const double*, int64_t, double);
template staging plan_staging<float>(const raft::handle_t&,
                                     const matrix<float, int64_t>&,
                                     const float*, const float*, row_selection,
                                     bool);
template staging plan_staging<double>(const raft::handle_t&,
                                      const matrix<double, int64_t>&,
                                      const double*, const double*,
                                      row_selection, bool);
template staged_problem<float> stage_problem<float>(
    const raft::handle_t&, const matrix<float, int64_t>&, const float*,
    const float*, const staging&, row_selection);
template staged_problem<double> stage_problem<double>(
    const raft::handle_t&, const matrix<double, int64_t>&, const double*,
    const double*, const staging&, row_selection);
template int64_t unstage_coef<float>(const raft::handle_t&, const float*,
                                     const staging&, int64_t, float*);
template int64_t unstage_coef<double>(const raft::handle_t&, const double*,
                                      const staging&, int64_t, double*);
template void require_finite_solution<float>(const float*, int64_t, float,
                                             glm_solver, rmm::cuda_stream_view);
template void require_finite_solution<double>(const double*, int64_t, double,
                                              glm_solver,
                                              rmm::cuda_stream_view);
template void require_stationary<float>(double, glm_solver);
template void require_stationary<double>(double, glm_solver);
template void require_in_range<float>(int64_t, float);
template void require_in_range<double>(int64_t, double);
template double stationarity_cosine<float>(const raft::handle_t&,
                                           const matrix<float, int64_t>&,
                                           const float*, const float*,
                                           row_selection, const staging&,
                                           const float*, bool, float);
template double stationarity_cosine<double>(const raft::handle_t&,
                                            const matrix<double, int64_t>&,
                                            const double*, const double*,
                                            row_selection, const staging&,
                                            const double*, bool, double);

}  // namespace cumlrkt::glm
