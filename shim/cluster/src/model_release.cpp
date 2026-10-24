#include <cstddef>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "cumlrkt/model.hpp"
#include "raftrkt/job.hpp"
#include "support.hpp"

extern "C" {

int cuml_cluster_model_release(void* handle, const rr_view* model) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    cumlrkt::cluster::in_device_scope(
        handle, {{"model", model}}, [&](const raft::handle_t& /*h*/) {
          static_cast<void>(cumlrkt::any_model(model, "model").release());
        });
  });
}

void cuml_cluster_model_release_job(void* args, size_t args_len,
                                    rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_model_release>(args, args_len, outcome,
                                                cumlrkt::last_error_slot());
}
}
