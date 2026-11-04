#include "glm_staging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <raft/core/handle.hpp>
#include <ranges>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
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

device_columns columns_of(const staging& s, rmm::cuda_stream_view stream) {
  std::vector<int> e(s.kept.size());
  std::ranges::transform(s.kept, e.begin(),
                         [&](int64_t j) { return s.column_exponent[j]; });
  return {to_device(s.kept, stream), to_device(e, stream)};
}

double cosine_of(const moment& m, scaled_sum y) {
  const double ratio =
      (m.cross.value * m.cross.value) / (m.square.value * y.value);
  return std::sqrt(std::ldexp(
      ratio, (2 * m.cross.exponent) - m.square.exponent - y.exponent));
}

}  // namespace

int even_floor(int x) {
  return x - (((x % 2) + 2) % 2);
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
bool constant_vector(const raft::handle_t& handle, const T* v, int64_t n,
                     const T* w, bool fit_intercept) {
  const std::vector<char> c = constant_columns(
      handle, matrix<T, int64_t>{const_cast<T*>(v), n, 1}, w, fit_intercept);
  return c[0] != 0;
}

template <typename T>
staging plan_staging(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const T* y, const T* w, bool fit_intercept,
                     exponents mode) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const std::vector<double> largest = column_largest(x, w, stream);
  std::vector<int> unit(x.d);
  std::ranges::transform(largest, unit.begin(), unit_exponent);
  const std::vector<double> spread =
      column_spread(x, w, unit, fit_intercept, stream);
  const std::vector<char> constant =
      constant_columns(handle, x, w, fit_intercept);
  staging s;
  s.column_exponent.resize(x.d);
  for (int64_t j = 0; j < x.d; ++j) {
    s.column_exponent[j] = spread_exponent<T>(unit[j], spread[j], x.n);
    if (constant[j] == 0) {
      s.kept.push_back(j);
    }
  }
  if (mode == exponents::uniform && !s.kept.empty()) {
    const int shared =
        std::ranges::min(s.kept | std::views::transform([&](int64_t j) {
                           return s.column_exponent[j];
                         }));
    std::ranges::fill(s.column_exponent, shared);
  }
  s.y_exponent = unit_exponent(largest_counted(y, x.n, w, stream));
  return s;
}

template <typename T>
buffer<T> staged_weights(const raft::handle_t& handle, const T* w, int64_t n) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<T> staged = scratch<T>(w == nullptr ? 0 : n, stream);
  if (w != nullptr) {
    const int exponent =
        even_floor(unit_exponent(largest_counted(w, n, w, stream)));
    stage_elements(handle, w, w, row_selection{nullptr, n}, exponent,
                   staged.data());
  }
  return staged;
}

template <typename T>
void stage_columns(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                   const T* w, const staging& s, row_selection rows, T* out) {
  const device_columns columns = columns_of(s, stream_of(handle));
  stage_matrix(handle, x, w, rows, columns.view(), out);
  settle(handle);
}

template <typename T>
void stage_targets(const raft::handle_t& handle, const T* y, const T* w,
                   const staging& s, row_selection rows, T* staged_y,
                   T* staged_w) {
  stage_elements(handle, y, w, rows, s.y_exponent, staged_y);
  if (w != nullptr) {
    stage_elements(handle, w, w, rows, 0, staged_w);
  }
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
                             int algorithm, rmm::cuda_stream_view stream) {
  if (first_non_finite(staged, d, stream) >= 0 || !std::isfinite(intercept)) {
    throw logic_error(
        std::format("X: the {} solve gave a non-finite coefficient",
                    algorithm_name(algorithm)));
  }
}

template <typename T>
double residual_cosine(const raft::handle_t& handle,
                       const matrix<T, int64_t>& x, const staging& s,
                       const T* y, const T* w, const T* coef,
                       bool fit_intercept) {
  const device_columns columns = columns_of(s, stream_of(handle));
  const residual_terms terms =
      residual_moments(handle, x, y, w, columns.view(), coef, fit_intercept);
  double worst = 0;
  for (size_t k = 0; k < terms.moments.size(); ++k) {
    if (terms.largest[k] == 0) {
      continue;
    }
    const double g = cosine_of(terms.moments[k], terms.y_square);
    worst = std::isnan(g) || std::isnan(worst) ? NAN : std::max(worst, g);
  }
  return worst;
}

template bool constant_vector<float>(const raft::handle_t&, const float*,
                                     int64_t, const float*, bool);
template bool constant_vector<double>(const raft::handle_t&, const double*,
                                      int64_t, const double*, bool);
template buffer<float> staged_weights<float>(const raft::handle_t&,
                                             const float*, int64_t);
template buffer<double> staged_weights<double>(const raft::handle_t&,
                                               const double*, int64_t);
template staging plan_staging<float>(const raft::handle_t&,
                                     const matrix<float, int64_t>&,
                                     const float*, const float*, bool,
                                     exponents);
template staging plan_staging<double>(const raft::handle_t&,
                                      const matrix<double, int64_t>&,
                                      const double*, const double*, bool,
                                      exponents);
template void stage_columns<float>(const raft::handle_t&,
                                   const matrix<float, int64_t>&, const float*,
                                   const staging&, row_selection, float*);
template void stage_columns<double>(const raft::handle_t&,
                                    const matrix<double, int64_t>&,
                                    const double*, const staging&,
                                    row_selection, double*);
template void stage_targets<float>(const raft::handle_t&, const float*,
                                   const float*, const staging&, row_selection,
                                   float*, float*);
template void stage_targets<double>(const raft::handle_t&, const double*,
                                    const double*, const staging&,
                                    row_selection, double*, double*);
template int64_t unstage_coef<float>(const raft::handle_t&, const float*,
                                     const staging&, int64_t, float*);
template int64_t unstage_coef<double>(const raft::handle_t&, const double*,
                                      const staging&, int64_t, double*);
template void require_finite_solution<float>(const float*, int64_t, float, int,
                                             rmm::cuda_stream_view);
template void require_finite_solution<double>(const double*, int64_t, double,
                                              int, rmm::cuda_stream_view);
template double residual_cosine<float>(const raft::handle_t&,
                                       const matrix<float, int64_t>&,
                                       const staging&, const float*,
                                       const float*, const float*, bool);
template double residual_cosine<double>(const raft::handle_t&,
                                        const matrix<double, int64_t>&,
                                        const staging&, const double*,
                                        const double*, const double*, bool);

}  // namespace cumlrkt::glm
