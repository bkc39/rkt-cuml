#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/metrics.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

namespace {

int unimplemented(void* handle) {
  return cumlrkt::guarded(handle, {}, [](const raft::handle_t&) {
    throw cumlrkt::logic_error("not implemented");
  });
}

}  // namespace

extern "C" {

int cuml_metrics_contingency_matrix(void* handle, const rr_view*,
                                    const rr_view*, rr_deposit*) {
  return unimplemented(handle);
}

int cuml_metrics_adjusted_rand_score(void* handle, const rr_view*,
                                     const rr_view*, double*) {
  return unimplemented(handle);
}

int cuml_metrics_silhouette_score(void* handle, const rr_view*,
                                  const rr_view*, int32_t, int64_t, double*) {
  return unimplemented(handle);
}

int cuml_metrics_silhouette_samples(void* handle, const rr_view*,
                                    const rr_view*, rr_deposit*, int32_t,
                                    int64_t, double*) {
  return unimplemented(handle);
}

int cuml_metrics_trustworthiness(void* handle, const rr_view*, const rr_view*,
                                 int64_t, int32_t, int64_t, double*) {
  return unimplemented(handle);
}

CUMLRKT_JOB(cuml_metrics_contingency_matrix)
CUMLRKT_JOB(cuml_metrics_adjusted_rand_score)
CUMLRKT_JOB(cuml_metrics_silhouette_score)
CUMLRKT_JOB(cuml_metrics_silhouette_samples)
CUMLRKT_JOB(cuml_metrics_trustworthiness)
}
