#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

extern "C" {

int cuml_cluster_single_linkage_fit(void* /*handle*/, const rr_view* /*x*/,
                                    rr_deposit* /*out*/, int64_t /*n_clusters*/,
                                    int32_t /*metric*/,
                                    int32_t /*connectivity*/, int32_t /*c*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

void cuml_cluster_single_linkage_fit_job(void* args, size_t args_len,
                                         rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_single_linkage_fit>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}
}
