#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
struct fit_result {
  std::unique_ptr<buffer<T>> coef;
  double intercept;
};

struct counted_rows {
  buffer<int64_t> index;
  row_selection rows;
};

struct fit_plan {
  staging s;
  bool fit_intercept;
  int algorithm;
  counted_rows counted;
};

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
fit_result<T> solved(const raft::handle_t& handle, const fit_inputs<T>& in,
                     const fit_plan& plan) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  const staging& s = plan.s;
  const auto kept = static_cast<int64_t>(s.kept.size());
  const row_selection rows = plan.counted.rows;
  require_workspace(
      plan.algorithm, rows.count, kept,
      cusolver_lwork_of<T>(handle, plan.algorithm, {rows.count, kept}));
  staged_problem<T> p = stage_problem(handle, x, y, w, s, rows);
  buffer<T> cs = scratch<T>(kept, stream);
  T intercept{0};
  ML::GLM::olsFit(handle, p.x.data(), static_cast<size_t>(rows.count),
                  static_cast<size_t>(kept), p.y.data(), cs.data(), &intercept,
                  plan.fit_intercept, plan.algorithm,
                  w == nullptr ? nullptr : p.w.data());
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, plan.algorithm, stream);
  require_least_squares<T>(
      residual_cosine(handle, x, y, w, rows, s, cs.data(), plan.fit_intercept),
      plan.algorithm);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  const T unscaled = std::ldexp(intercept, -s.y_exponent);
  require_in_range(unstage_coef(handle, cs.data(), s, x.d, coef->data()),
                   unscaled);
  return {std::move(coef),
          plan.fit_intercept ? static_cast<double>(unscaled) : 0.0};
}

template <typename T>
fit_plan plan_fit(const raft::handle_t& handle, const fit_inputs<T>& in,
                  const fit_request& request) {
  const auto& [x, y, w] = in;
  const rmm::cuda_stream_view stream = stream_of(handle);
  const bool intercepted = request.fit_intercept == 1;
  const int64_t positive = positive_count(w, x.n, stream);
  counted_rows counted = counted_rows_of(w, x.n, positive, stream);
  staging planned = plan_staging(handle, x, y, w, counted.rows, intercepted);
  const auto kept = static_cast<int64_t>(planned.kept.size());
  const bool wide = kept > positive - (intercepted ? 1 : 0);
  const int algorithm =
      wide || kept == 1 ? CUML_LINEAR_MODEL_OLS_SVD : request.algorithm;
  return {wide ? uniform(std::move(planned)) : std::move(planned), intercepted,
          algorithm, std::move(counted)};
}

template <typename T>
double fit_into(const raft::handle_t& handle, const fit_request& request,
                rr_deposit* out) {
  const checked_fit<T> checked = checked_inputs<T>(handle, request);
  const fit_inputs<T>& raw = checked.in;
  const buffer<T> weights =
      staged_weights(handle, raw.w, raw.x.n, checked.largest_weight);
  const fit_inputs<T> in{raw.x, raw.y,
                         raw.w == nullptr ? nullptr : weights.data()};
  const fit_plan plan = plan_fit(handle, in, request);
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, plan.fit_intercept);
  fit_result<T> r = constant_y || plan.s.kept.empty()
                        ? nothing_to_solve(handle, in, plan, constant_y)
                        : solved(handle, in, plan);
  settle(handle);
  const rr_view region = raftrkt::region(*r.coef);
  raftrkt::deposit(out, "out", handle, std::move(r.coef), {region});
  return r.intercept;
}

}  // namespace

}  // namespace cumlrkt::glm

extern "C" {

int cuml_linear_model_linear_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, rr_deposit* out, int32_t fit_intercept,
    int32_t algorithm, double* intercept) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const icpt = raftrkt::require(intercept, "intercept");
        const cumlrkt::glm::fit_request request{x, y, sample_weight,
                                                fit_intercept, algorithm};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          *icpt = cumlrkt::glm::fit_into<decltype(tag)>(h, request, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_linear_regression_fit)
}
