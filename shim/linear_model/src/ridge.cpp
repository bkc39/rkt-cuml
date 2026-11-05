#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_fit.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "ridge_staging.hpp"

namespace cumlrkt::glm {

namespace {

struct ridge_request {
  fit_request fit;
  double alpha;
};

template <typename T>
int64_t ridge_lwork_of(const raft::handle_t& handle, int algorithm,
                       ols_shape shape) {
  const auto [m, d] = shape;
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  int lwork = 0;
  const cusolverStatus_t status =
      algorithm == CUML_LINEAR_MODEL_RIDGE_SVD
          ? raft::linalg::detail::cusolverDngesvd_bufferSize<T>(
                solver, static_cast<int>(m), static_cast<int>(d), &lwork)
          : raft::linalg::detail::cusolverDnsyevd_bufferSize<T>(
                solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                static_cast<int>(d), nullptr, static_cast<int>(d), nullptr,
                &lwork);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw logic_error(
        std::format("X: {}x{} elements are past cuSOLVER's {} workspace query",
                    m, d, algorithm_name(algorithm)));
  }
  return lwork;
}

template <typename T>
fit_result<T> ridge_solved(const raft::handle_t& handle,
                           const fit_inputs<T>& in, const fit_plan& plan,
                           T alpha) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  const staging& s = plan.s;
  const auto kept = static_cast<int64_t>(s.kept.size());
  const row_selection rows = plan.counted.rows;
  require_ridge_workspace(
      plan.algorithm, {rows.count, kept},
      ridge_lwork_of<T>(handle, plan.algorithm, {rows.count, kept}));
  staged_problem<T> p = stage_problem(handle, x, y, w, s, rows);
  buffer<T> cs = scratch<T>(kept, stream);
  T intercept{0};
  T penalty = alpha;
  ML::GLM::ridgeFit(handle, p.x.data(), static_cast<size_t>(rows.count),
                    static_cast<size_t>(kept), p.y.data(), &penalty, 1,
                    cs.data(), &intercept, plan.fit_intercept, plan.algorithm,
                    w == nullptr ? nullptr : p.w.data());
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, plan.algorithm, stream);
  require_ridge_solution<T>(
      stationarity_cosine(handle, x, y, w, rows, s, cs.data(),
                          plan.fit_intercept, alpha),
      plan.algorithm);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  const T unscaled = std::ldexp(intercept, -s.y_exponent);
  require_in_range(unstage_coef(handle, cs.data(), s, x.d, coef->data()),
                   unscaled);
  return {std::move(coef),
          plan.fit_intercept ? static_cast<double>(unscaled) : 0.0};
}

template <typename T>
double ridge_fit_into(const raft::handle_t& handle,
                      const ridge_request& request, rr_deposit* out) {
  const fit_request& fit = request.fit;
  const checked_fit<T> checked = checked_inputs<T>(handle, fit, [&] {
    require_ridge_codes({fit.algorithm, fit.fit_intercept, request.alpha});
  });
  const fit_inputs<T>& raw = checked.in;
  const rmm::cuda_stream_view stream = stream_of(handle);
  const buffer<T> weights =
      staged_weights(handle, raw.w, raw.x.n, checked.largest_weight);
  const fit_inputs<T> in{raw.x, raw.y,
                         raw.w == nullptr ? nullptr : weights.data()};
  const bool intercepted = fit.fit_intercept == 1;
  const int64_t positive = positive_count(in.w, in.x.n, stream);
  counted_rows counted = counted_rows_of(in.w, in.x.n, positive, stream);
  const int weight_power =
      raw.w == nullptr ? 0 : even_floor(unit_exponent(checked.largest_weight));
  ridge_stage<T> staged = ridge_staging<T>(
      plan_staging(handle, in.x, in.y, in.w, counted.rows, intercepted),
      counted.rows.count, {request.alpha, weight_power});
  if (staged.alpha == 0) {
    return ols_fit_into<T>(handle,
                           {fit.x, fit.y, fit.w, fit.fit_intercept,
                            fit.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD
                                ? CUML_LINEAR_MODEL_OLS_SVD_QR
                                : CUML_LINEAR_MODEL_OLS_EIG},
                           out);
  }
  const auto kept = static_cast<int64_t>(staged.s.kept.size());
  const fit_plan plan{std::move(staged.s), intercepted,
                      ridge_algorithm(fit.algorithm, positive, kept),
                      std::move(counted)};
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, intercepted);
  return deposit_fit(handle,
                     constant_y || kept == 0
                         ? nothing_to_solve(handle, in, plan, constant_y)
                         : ridge_solved(handle, in, plan, staged.alpha),
                     out);
}

}  // namespace

}  // namespace cumlrkt::glm

extern "C" {

int cuml_linear_model_ridge_fit(void* handle, const rr_view* x,
                                const rr_view* y, const rr_view* sample_weight,
                                rr_deposit* out, double alpha,
                                int32_t fit_intercept, int32_t solver,
                                double* intercept) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const icpt = raftrkt::require(intercept, "intercept");
        const cumlrkt::glm::ridge_request request{
            {x, y, sample_weight, fit_intercept, solver}, alpha};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          *icpt = cumlrkt::glm::ridge_fit_into<decltype(tag)>(h, request, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_ridge_fit)
}
