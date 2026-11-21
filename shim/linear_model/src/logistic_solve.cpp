#include "logistic_solve.hpp"

#include <cuml/linear_model/qn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <initializer_list>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
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
bool all_finite(const buffer<T>& params, rmm::cuda_stream_view stream) {
  return first_non_finite(params.data(), static_cast<int64_t>(params.size()),
                          stream) < 0;
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
std::vector<T> mapped_forward(const std::vector<T>& plain,
                              const rescue_plan& plan, int shared,
                              const qn_shape& shape) {
  const int64_t k = shape.k;
  std::vector<T> out(plain.size());
  for (int64_t c = 0; c < k; ++c) {
    double intercept = shape.fit_intercept
                           ? static_cast<double>(plain[(shape.p * k) + c])
                           : 0.0;
    for (int64_t q = 0; q < shape.p; ++q) {
      const auto v = static_cast<double>(plain[(q * k) + c]);
      out[(q * k) + c] = static_cast<T>(std::ldexp(v, shared - plan.shift[q]));
      intercept += std::ldexp(v, shared) * plan.mean[q];
    }
    if (shape.fit_intercept) {
      out[(shape.p * k) + c] = static_cast<T>(intercept);
    }
  }
  return out;
}

template <typename T>
std::vector<T> mapped_back(const std::vector<T>& rescue,
                           const rescue_plan& plan, int shared,
                           const qn_shape& shape) {
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
qn_solution<T> qn_solved(const raft::handle_t& handle, T* x, T* y, T* w,
                         const qn_shape& shape, penalties pen,
                         const fit_options& options,
                         const std::vector<T>& start) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<T> params =
      start.empty()
          ? zeroed<T>((shape.p + (shape.fit_intercept ? 1 : 0)) * shape.k,
                      stream)
          : to_device(start, stream);
  T objective{0};
  int n_iter = 0;
  ML::GLM::qnFit<T, int>(
      handle, qn_params_of(options, shape.classes, pen), x, false, y,
      static_cast<int>(shape.m), static_cast<int>(shape.p),
      static_cast<int>(shape.classes), params.data(), &objective, &n_iter, w);
  settle(handle);
  return {std::move(params), n_iter};
}

template <typename T>
attempt<T> attempted(const raft::handle_t& handle, T* x, T* y, T* w,
                     solve_check check, const fit_options& options,
                     const std::vector<T>& start) {
  qn_solution<T> solution =
      qn_solved<T>(handle, x, y, w, check.shape, check.pen, options, start);
  require_finite_solution(solution.params.data(),
                          static_cast<int64_t>(solution.params.size()),
                          stream_of(handle));
  check.n_iter = solution.n_iter;
  const resolution r =
      checked<T>(handle, x, y, w, solution.params, options, check);
  return {std::move(solution), r, std::move(check)};
}

template <typename T>
rescue<T> rescued(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  row_selection rows, const column_stats& stats,
                  staged_fit<T>& st, const attempt<T>& plain,
                  const fit_options& options, const qn_solver<T>& solve) {
  const bool per_column = options.penalty == CUML_LINEAR_MODEL_PENALTY_NONE;
  if ((!options.fit_intercept && !per_column) ||
      plain.solution.n_iter >= options.max_iter) {
    return {std::nullopt, plain.solution.n_iter, std::nullopt};
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  const qn_shape& shape = st.shape;
  const rescue_plan plan =
      plan_of<T>(stats, st.kept, st.shift, per_column, shape.m);
  const std::vector<T> cold;
  const std::vector<T> warm = mapped_forward(
      to_host(plain.solution.params, stream), plan, st.shift, shape);
  buffer<T> copy = scratch<T>(shape.m * shape.p, stream);
  {
    const buffer<int64_t> kept = to_device(st.kept, stream);
    const buffer<int> shifts = to_device(plan.shift, stream);
    const buffer<double> means = to_device(plan.mean, stream);
    stage_centred_rows<T>(handle, x, rows, kept.data(), shape.p, shifts.data(),
                          means.data(), copy.data());
  }
  solve_check total = plain.check;
  std::optional<resolution> last;
  for (const std::vector<T>* start : {&cold, &warm}) {
    fit_options budget = options;
    budget.max_iter = options.max_iter - total.n_iter;
    if (budget.max_iter < 1) {
      break;
    }
    const qn_solution<T> tried =
        solve(handle, copy.data(), st.y.data(), st.weights(), shape,
              plain.check.pen, budget, *start);
    total.n_iter += tried.n_iter;
    if (!all_finite(tried.params, stream)) {
      continue;
    }
    buffer<T> params = to_device(
        mapped_back(to_host(tried.params, stream), plan, st.shift, shape),
        stream);
    if (!all_finite(params, stream)) {
      continue;
    }
    last = checked<T>(handle, st.x(), st.y.data(), st.weights(), params,
                      options, total);
    if (last->resolved()) {
      return {qn_solution<T>{std::move(params), total.n_iter}, total.n_iter,
              last};
    }
  }
  return {std::nullopt, total.n_iter, last};
}

template qn_solution<float> qn_solved<float>(const raft::handle_t&, float*,
                                             float*, float*, const qn_shape&,
                                             penalties, const fit_options&,
                                             const std::vector<float>&);
template attempt<float> attempted<float>(const raft::handle_t&, float*, float*,
                                         float*, solve_check,
                                         const fit_options&,
                                         const std::vector<float>&);
template rescue<float> rescued<float>(const raft::handle_t&,
                                      const matrix<float, int64_t>&,
                                      row_selection, const column_stats&,
                                      staged_fit<float>&, const attempt<float>&,
                                      const fit_options&,
                                      const qn_solver<float>&);
template qn_solution<double> qn_solved<double>(const raft::handle_t&, double*,
                                               double*, double*,
                                               const qn_shape&, penalties,
                                               const fit_options&,
                                               const std::vector<double>&);
template attempt<double> attempted<double>(const raft::handle_t&, double*,
                                           double*, double*, solve_check,
                                           const fit_options&,
                                           const std::vector<double>&);
template rescue<double> rescued<double>(
    const raft::handle_t&, const matrix<double, int64_t>&, row_selection,
    const column_stats&, staged_fit<double>&, const attempt<double>&,
    const fit_options&, const qn_solver<double>&);

}  // namespace cumlrkt::glm::logistic
