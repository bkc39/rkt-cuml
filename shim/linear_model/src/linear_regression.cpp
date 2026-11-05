#include <cstdint>
#include <raft/core/handle.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_fit.hpp"
#include "glm_staging.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
fit_plan plan_fit(prepared_fit<T>& prepared, int algorithm) {
  const auto kept = static_cast<int64_t>(prepared.planned.kept.size());
  const bool wide =
      kept > prepared.positive - (prepared.fit_intercept ? 1 : 0);
  return {wide ? uniform(std::move(prepared.planned))
               : std::move(prepared.planned),
          prepared.fit_intercept,
          {glm_family::ols,
           wide || kept == 1 ? CUML_LINEAR_MODEL_OLS_SVD : algorithm},
          std::move(prepared.counted)};
}

}  // namespace

template <typename T>
double ols_fit_prepared(const raft::handle_t& handle, prepared_fit<T> prepared,
                        int algorithm, rr_deposit* out) {
  const fit_inputs<T> in = prepared.in;
  const fit_plan plan = plan_fit(prepared, algorithm);
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, plan.fit_intercept);
  return deposit_fit(handle,
                     constant_y || plan.s.kept.empty()
                         ? nothing_to_solve(handle, in, plan, constant_y)
                         : solved(handle, in, plan, T{0}),
                     out);
}

template <typename T>
double ols_fit_into(const raft::handle_t& handle, const fit_request& request,
                    rr_deposit* out) {
  const checked_fit<T> checked =
      checked_inputs<T>(handle, request, [&] { require_ols_codes(request); });
  return ols_fit_prepared(
      handle, prepare_fit(handle, checked, request.fit_intercept),
      request.algorithm, out);
}

template double ols_fit_prepared<float>(const raft::handle_t&,
                                        prepared_fit<float>, int, rr_deposit*);
template double ols_fit_prepared<double>(const raft::handle_t&,
                                         prepared_fit<double>, int,
                                         rr_deposit*);
template double ols_fit_into<float>(const raft::handle_t&, const fit_request&,
                                    rr_deposit*);
template double ols_fit_into<double>(const raft::handle_t&, const fit_request&,
                                     rr_deposit*);

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
          *icpt = cumlrkt::glm::ols_fit_into<decltype(tag)>(h, request, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_linear_regression_fit)
}
