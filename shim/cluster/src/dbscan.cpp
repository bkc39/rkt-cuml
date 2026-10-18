#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

extern "C" {

int cuml_cluster_dbscan_fit(void* /*handle*/, const rr_view* /*x*/,
                            const rr_view* /*sample_weight*/,
                            rr_deposit* /*out*/, double /*eps*/,
                            int32_t /*min_samples*/, int32_t /*metric*/,
                            int32_t /*out_dtype*/,
                            int64_t /*max_mbytes_per_batch*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

void cuml_cluster_dbscan_fit_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_dbscan_fit>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}
}
