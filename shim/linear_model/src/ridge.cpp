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
#include "ridge_staging.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
double ridge_fit_into(const raft::handle_t& handle,
                      const ridge_request& request, rr_deposit* out) {
  const fit_request& fit = request.fit;
  const checked_fit<T> checked =
      checked_inputs<T>(handle, fit, [&] { require_ridge_codes(request); });
  prepared_fit<T> prepared = prepare_fit(handle, checked, fit.fit_intercept);
  const ridge_scale<T> scale =
      ridge_scale_of<T>(prepared.planned, prepared.counted.rows.count,
                        {request.alpha, prepared.weight_power});
  if (scale.negligible) {
    return ols_fit_prepared(handle, std::move(prepared),
                            lstsq_algorithm(fit.algorithm), out);
  }
  const fit_inputs<T> in = prepared.in;
  const auto kept = static_cast<int64_t>(prepared.planned.kept.size());
  const fit_plan plan{shared_at(std::move(prepared.planned), scale.shared),
                      prepared.fit_intercept,
                      {glm_family::ridge,
                       ridge_algorithm(fit.algorithm, prepared.positive, kept)},
                      std::move(prepared.counted)};
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, plan.fit_intercept);
  return deposit_fit(handle,
                     constant_y ? nothing_to_solve(handle, in, plan, constant_y)
                                : solved(handle, in, plan, scale.alpha),
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
