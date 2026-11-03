#include <raft/core/handle.hpp>

#include "cumlrkt/metrics.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;

void unimplemented() {
  throw logic_error("not implemented");
}

}  // namespace

extern "C" {

int cuml_metrics_r2_score(void* handle, const rr_view* y_true,
                          const rr_view* y_pred, const rr_view* sample_weight,
                          double* score) {
  (void)score;
  return guarded(handle,
                 {{"y-true", y_true},
                  {"y-pred", y_pred},
                  {"sample-weight", sample_weight}},
                 [&](const raft::handle_t&) { unimplemented(); });
}

int cuml_metrics_mean_squared_error(void* handle, const rr_view* y_true,
                                    const rr_view* y_pred,
                                    const rr_view* sample_weight,
                                    double* error) {
  (void)error;
  return guarded(handle,
                 {{"y-true", y_true},
                  {"y-pred", y_pred},
                  {"sample-weight", sample_weight}},
                 [&](const raft::handle_t&) { unimplemented(); });
}

int cuml_metrics_accuracy_score(void* handle, const rr_view* y_true,
                                const rr_view* y_pred,
                                const rr_view* sample_weight, double* correct,
                                double* total) {
  (void)correct;
  (void)total;
  return guarded(handle,
                 {{"y-true", y_true},
                  {"y-pred", y_pred},
                  {"sample-weight", sample_weight}},
                 [&](const raft::handle_t&) { unimplemented(); });
}

CUMLRKT_JOB(cuml_metrics_r2_score)
CUMLRKT_JOB(cuml_metrics_mean_squared_error)
CUMLRKT_JOB(cuml_metrics_accuracy_score)
}
