#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

extern "C" {

int cuml_linear_model_ridge_fit(void* handle, const rr_view* x,
                                const rr_view* y, const rr_view* sample_weight,
                                rr_deposit* out, double alpha,
                                int32_t fit_intercept, int32_t solver,
                                double* intercept) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t&) {
        static_cast<void>(out);
        static_cast<void>(alpha);
        static_cast<void>(fit_intercept);
        static_cast<void>(solver);
        static_cast<void>(intercept);
        throw cumlrkt::logic_error("not implemented");
      });
}

CUMLRKT_JOB(cuml_linear_model_ridge_fit)
}
