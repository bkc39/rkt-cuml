#include <cstdint>

#include "cumlrkt/decomposition.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

namespace {

int unimplemented(void* handle) {
  return cumlrkt::guarded(handle, {}, [](const raft::handle_t&) {
    throw cumlrkt::logic_error("pca: not implemented");
  });
}

}  // namespace

extern "C" {

int cuml_decomposition_pca_fit(void* handle, const rr_view* /*x*/,
                               rr_deposit* /*out*/, int32_t /*all_components*/,
                               int64_t /*n_components*/, int32_t /*solver*/,
                               double /*tol*/, int64_t /*iterations*/,
                               double* /*noise_variance*/,
                               int64_t* /*n_samples*/) {
  return unimplemented(handle);
}

int cuml_decomposition_pca_transform(
    void* handle, const rr_view* /*x*/, const rr_view* /*components*/,
    const rr_view* /*mean*/, const rr_view* /*explained_variance*/,
    rr_deposit* /*out*/, int32_t /*whiten*/) {
  return unimplemented(handle);
}

int cuml_decomposition_pca_inverse_transform(
    void* handle, const rr_view* /*x*/, const rr_view* /*components*/,
    const rr_view* /*mean*/, const rr_view* /*explained_variance*/,
    rr_deposit* /*out*/, int32_t /*whiten*/) {
  return unimplemented(handle);
}

CUMLRKT_JOB(cuml_decomposition_pca_fit)
CUMLRKT_JOB(cuml_decomposition_pca_transform)
CUMLRKT_JOB(cuml_decomposition_pca_inverse_transform)
}
