#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/decomposition.h"
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

int cuml_decomposition_tsvd_fit(void* handle, const rr_view*, rr_deposit*,
                                int64_t, int32_t, double, int64_t, int32_t) {
  return unimplemented(handle);
}

int cuml_decomposition_tsvd_transform(void* handle, const rr_view*,
                                      const rr_view*, rr_deposit*) {
  return unimplemented(handle);
}

int cuml_decomposition_tsvd_inverse_transform(void* handle, const rr_view*,
                                              const rr_view*, rr_deposit*) {
  return unimplemented(handle);
}

CUMLRKT_JOB(cuml_decomposition_tsvd_fit)
CUMLRKT_JOB(cuml_decomposition_tsvd_transform)
CUMLRKT_JOB(cuml_decomposition_tsvd_inverse_transform)
}
