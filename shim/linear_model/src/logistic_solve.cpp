#include "logistic_solve.hpp"

#include <cuml/linear_model/qn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "logistic_checks.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "logistic_resolve.hpp"

namespace cumlrkt::glm::logistic {

namespace {

constexpr double change_tol_per_tol = 0.01;

ML::GLM::qn_params qn_params_of(const fit_options& o, int64_t k, penalties p) {
  ML::GLM::qn_params q;
  q.loss = k == 2 ? ML::GLM::QN_LOSS_LOGISTIC : ML::GLM::QN_LOSS_SOFTMAX;
  q.penalty_l1 = p.l1;
  q.penalty_l2 = p.l2;
  q.grad_tol = o.tol;
  q.change_tol = o.tol * change_tol_per_tol;
  q.max_iter = o.max_iter;
  q.linesearch_max_iter = o.linesearch_max_iter;
  q.lbfgs_memory = o.lbfgs_memory;
  q.verbose = quiet_level;
  q.fit_intercept = o.fit_intercept;
  q.penalty_normalized = o.normalized;
  return q;
}

template <typename T>
resolution checked(const raft::handle_t& handle, const T* x, const T* y,
                   const T* w, const buffer<T>& params,
                   const fit_options& options, const solve_check& check) {
  const std::vector<T> host = to_host(params, stream_of(handle));
  return resolution_of(
      residuals_of<T>(handle, x, y, w, params.data(), check.shape),
      std::vector<double>(host.begin(), host.end()), options, check);
}

struct rescue_plan {
  std::vector<int> shift;
  std::vector<double> mean;
};

template <typename T>
rescue_plan plan_of(const column_stats& stats, const std::vector<int64_t>& kept,
                    int shared, bool per_column, int64_t m) {
  rescue_plan plan{std::vector<int>(kept.size(), shared),
                   std::vector<double>(kept.size(), 0.0)};
  for (std::size_t q = 0; q < kept.size(); ++q) {
    const int64_t j = kept[q];
    plan.mean[q] = stats.mean[j];
    if (per_column) {
      const int cap = column_reach<T>(m) - std::ilogb(stats.magnitude[j]) - 1;
      plan.shift[q] = std::min(-std::ilogb(stats.sigma[j]), cap);
    }
  }
  return plan;
}

template <typename T>
std::vector<T> mapped_back(const std::vector<T>& rescue, const rescue_plan& plan,
                           int shared, const qn_shape& shape) {
  const int64_t k = shape.k;
  std::vector<T> out(rescue.size());
  for (int64_t c = 0; c < k; ++c) {
    double intercept = shape.fit_intercept
                           ? static_cast<double>(rescue[(shape.p * k) + c])
                           : 0.0;
    for (int64_t q = 0; q < shape.p; ++q) {
      const auto v = static_cast<double>(rescue[(q * k) + c]);
      out[(q * k) + c] = static_cast<T>(std::ldexp(v, plan.shift[q] - shared));
      intercept -= std::ldexp(v, plan.shift[q]) * plan.mean[q];
    }
    if (shape.fit_intercept) {
      out[(shape.p * k) + c] = static_cast<T>(intercept);
    }
  }
  return out;
}

}  // namespace

template <typename T>
attempt<T> attempted(const raft::handle_t& handle, T* x, T* y, T* w,
                     solve_check check, const fit_options& options) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const qn_shape& shape = check.shape;
  buffer<T> params =
      zeroed<T>((shape.p + (shape.fit_intercept ? 1 : 0)) * shape.k, stream);
  T objective{0};
  int n_iter = 0;
  ML::GLM::qnFit<T, int>(
      handle, qn_params_of(options, shape.classes, check.pen), x, false, y,
      static_cast<int>(shape.m), static_cast<int>(shape.p),
      static_cast<int>(shape.classes), params.data(), &objective, &n_iter, w);
  settle(handle);
  require_finite_solution(params.data(), static_cast<int64_t>(params.size()),
                          stream);
  check.n_iter = n_iter;
  const resolution r = checked<T>(handle, x, y, w, params, options, check);
  return {{std::move(params), n_iter}, r, std::move(check)};
}

template <typename T>
std::optional<qn_solution<T>> rescued(const raft::handle_t& handle,
                                      const matrix<T, int64_t>& x,
                                      row_selection rows,
                                      const column_stats& stats,
                                      staged_fit<T>& st,
                                      const attempt<T>& plain,
                                      const fit_options& options) {
  const bool per_column = options.penalty == CUML_LINEAR_MODEL_PENALTY_NONE;
  if (!options.fit_intercept && !per_column) {
    return std::nullopt;
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  const qn_shape& shape = st.shape;
  const rescue_plan plan =
      plan_of<T>(stats, st.kept, st.shift, per_column, shape.m);
  solve_check check = plain.check;
  for (int64_t q = 0; q < shape.p; ++q) {
    check.sigma[q] = std::ldexp(check.sigma[q], plan.shift[q] - st.shift);
    check.mean[q] = 0.0;
  }
  std::optional<attempt<T>> rescue;
  {
    buffer<T> copy = scratch<T>(shape.m * shape.p, stream);
    const buffer<int64_t> kept = to_device(st.kept, stream);
    const buffer<int> shifts = to_device(plan.shift, stream);
    const buffer<double> means = to_device(plan.mean, stream);
    stage_centred_rows<T>(handle, x, rows, kept.data(), shape.p, shifts.data(),
                          means.data(), copy.data());
    rescue.emplace(attempted<T>(handle, copy.data(), st.y.data(), st.weights(),
                                check, options));
  }
  const std::vector<T> back =
      mapped_back(to_host(rescue->solution.params, stream), plan, st.shift,
                  shape);
  buffer<T> params = to_device(back, stream);
  require_finite_solution(params.data(), static_cast<int64_t>(params.size()),
                          stream);
  solve_check total = plain.check;
  total.n_iter += rescue->solution.n_iter;
  const resolution r = checked<T>(handle, st.x(), st.y.data(), st.weights(),
                                  params, options, total);
  if (!r.resolved()) {
    return std::nullopt;
  }
  return qn_solution<T>{std::move(params), total.n_iter};
}

#define CUMLRKT_LOGISTIC_SOLVE(T)                                            \
  template attempt<T> attempted<T>(const raft::handle_t&, T*, T*, T*,        \
                                   solve_check, const fit_options&);         \
  template std::optional<qn_solution<T>> rescued<T>(                         \
      const raft::handle_t&, const matrix<T, int64_t>&, row_selection,       \
      const column_stats&, staged_fit<T>&, const attempt<T>&,                \
      const fit_options&);

CUMLRKT_LOGISTIC_SOLVE(float)
CUMLRKT_LOGISTIC_SOLVE(double)

}  // namespace cumlrkt::glm::logistic
