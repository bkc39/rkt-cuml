#include <cstddef>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "cumlrkt/model.hpp"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

extern "C" {

int cuml_cluster_model_release(void* handle, const rr_view* model) {
  return cumlrkt::guarded(
      handle, {{"model", model}}, [&](const raft::handle_t& /*h*/) {
        static_cast<void>(cumlrkt::any_model(model, "model").release());
      });
}

CUMLRKT_JOB(cuml_cluster_model_release)
}
