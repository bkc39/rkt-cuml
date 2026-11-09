#include "glm_fit.hpp"

#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
std::unique_ptr<buffer<T>> zero_coef(int64_t d, rmm::cuda_stream_view stream) {
  auto coef = std::make_unique<buffer<T>>(scratch<T>(d, stream));
  raftrkt::cuda_check(cudaMemsetAsync(coef->data(), 0, coef->size() * sizeof(T),
                                      stream.value()),
                      "cudaMemsetAsync");
  return coef;
}

template <typename T>
counted_rows counted_rows_of(const T* w, int64_t n, int64_t positive,
                             rmm::cuda_stream_view stream) {
  const bool compact = positive < n && positive >= 2;
  buffer<int64_t> index = scratch<int64_t>(compact ? positive : 0, stream);
  if (compact) {
    positive_rows(w, n, index.data(), stream);
  }
  const row_selection rows{compact ? index.data() : nullptr,
                           compact ? positive : n};
  return {std::move(index), rows};
}

template <typename T>
fit_result<T> solved(const raft::handle_t& handle, const fit_inputs<T>& in,
                     const fit_plan& plan, T alpha) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  const staging& s = plan.s;
  const auto kept = static_cast<int64_t>(s.kept.size());
  const row_selection rows = plan.counted.rows;
  require_workspace(
      plan.solver, {rows.count, kept},
      cusolver_lwork_of<T>(handle, plan.solver, {rows.count, kept}));
  staged_problem<T> p = stage_problem(handle, x, y, w, s, rows);
  buffer<T> cs = scratch<T>(kept, stream);
  T intercept{0};
  const auto n = static_cast<size_t>(rows.count);
  const auto d = static_cast<size_t>(kept);
  T* const staged_w = w == nullptr ? nullptr : p.w.data();
  if (plan.solver.family == glm_family::ridge) {
    ML::GLM::ridgeFit(handle, p.x.data(), n, d, p.y.data(), &alpha, 1,
                      cs.data(), &intercept, plan.fit_intercept,
                      plan.solver.algorithm, staged_w);
  } else {
    ML::GLM::olsFit(handle, p.x.data(), n, d, p.y.data(), cs.data(), &intercept,
                    plan.fit_intercept, plan.solver.algorithm, staged_w);
  }
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, plan.solver.algorithm,
                          stream);
  require_stationary<T>(stationarity_cosine(handle, x, y, w, rows, s, cs.data(),
                                            plan.fit_intercept, alpha),
                        plan.solver);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  const T unscaled = std::ldexp(intercept, -s.y_exponent);
  require_in_range(unstage_coef(handle, cs.data(), s, x.d, coef->data()),
                   unscaled);
  return {std::move(coef),
          plan.fit_intercept ? static_cast<double>(unscaled) : 0.0};
}

}  // namespace

template <typename T>
fit_result<T> nothing_to_solve(const raft::handle_t& handle,
                               const fit_inputs<T>& in, const fit_plan& plan,
                               bool constant_y) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  auto coef = zero_coef<T>(x.d, stream);
  double intercept = 0;
  if (plan.fit_intercept && constant_y) {
    intercept = static_cast<double>(
        read_one(y, first_positive(w, x.n, stream), stream));
  } else if (plan.fit_intercept) {
    const centre c =
        target_centre(y, w, plan.counted.rows, plan.s.y_exponent, stream);
    intercept = static_cast<double>(static_cast<T>(
        std::ldexp(static_cast<T>(c.mean + c.remainder), -plan.s.y_exponent)));
  }
  settle(handle);
  return {std::move(coef), intercept};
}

template <typename T>
double deposit_fit(const raft::handle_t& handle, fit_result<T> r,
                   rr_deposit* out) {
  settle(handle);
  const rr_view region = raftrkt::region(*r.coef);
  raftrkt::deposit(out, "out", handle, std::move(r.coef), {region});
  return r.intercept;
}

template <typename T>
prepared_fit<T> prepare_fit(const raft::handle_t& handle,
                            const checked_fit<T>& checked,
                            int32_t fit_intercept) {
  const fit_inputs<T>& raw = checked.in;
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<T> weights =
      staged_weights(handle, raw.w, raw.x.n, checked.largest_weight);
  const fit_inputs<T> in{raw.x, raw.y,
                         raw.w == nullptr ? nullptr : weights.data()};
  const bool intercepted = fit_intercept == 1;
  const int64_t positive = positive_count(in.w, in.x.n, stream);
  counted_rows counted = counted_rows_of(in.w, in.x.n, positive, stream);
  staging planned =
      plan_staging(handle, in.x, in.y, in.w, counted.rows, intercepted);
  const int weight_power =
      raw.w == nullptr ? 0 : even_floor(unit_exponent(checked.largest_weight));
  return {std::move(weights), in,       intercepted,
          weight_power,       positive, std::move(counted),
          std::move(planned)};
}

template <typename T>
double finish_fit(const raft::handle_t& handle, const fit_inputs<T>& in,
                  const fit_plan& plan, T alpha, rr_deposit* out) {
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, plan.fit_intercept);
  return deposit_fit(handle,
                     constant_y || plan.s.kept.empty()
                         ? nothing_to_solve(handle, in, plan, constant_y)
                         : solved(handle, in, plan, alpha),
                     out);
}

template fit_result<float> nothing_to_solve<float>(const raft::handle_t&,
                                                   const fit_inputs<float>&,
                                                   const fit_plan&, bool);
template fit_result<double> nothing_to_solve<double>(const raft::handle_t&,
                                                     const fit_inputs<double>&,
                                                     const fit_plan&, bool);
template double deposit_fit<float>(const raft::handle_t&, fit_result<float>,
                                   rr_deposit*);
template double deposit_fit<double>(const raft::handle_t&, fit_result<double>,
                                    rr_deposit*);
template prepared_fit<float> prepare_fit<float>(const raft::handle_t&,
                                                const checked_fit<float>&,
                                                int32_t);
template prepared_fit<double> prepare_fit<double>(const raft::handle_t&,
                                                  const checked_fit<double>&,
                                                  int32_t);
template double finish_fit<float>(const raft::handle_t&,
                                  const fit_inputs<float>&, const fit_plan&,
                                  float, rr_deposit*);
template double finish_fit<double>(const raft::handle_t&,
                                   const fit_inputs<double>&, const fit_plan&,
                                   double, rr_deposit*);

}  // namespace cumlrkt::glm
