#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/preprocessing.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;

[[noreturn]] void unimplemented() {
  throw logic_error("standard-scaler: not implemented");
}

}  // namespace

extern "C" {

int cuml_preprocessing_standard_scaler_fit(void* handle, const rr_view* x,
                                           rr_deposit* /*out*/,
                                           int32_t /*with_mean*/,
                                           int32_t /*with_std*/,
                                           int64_t /*scratch_elements*/,
                                           int64_t* /*n_samples*/) {
  return guarded(handle, {{"X", x}},
                 [&](const raft::handle_t& /*h*/) { unimplemented(); });
}

int cuml_preprocessing_standard_scaler_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* /*out*/, int64_t /*n_features*/) {
  return guarded(handle, {{"X", x}, {"mean", mean}, {"scale", scale}},
                 [&](const raft::handle_t& /*h*/) { unimplemented(); });
}

int cuml_preprocessing_standard_scaler_inverse_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* /*out*/, int64_t /*n_features*/) {
  return guarded(handle, {{"X", x}, {"mean", mean}, {"scale", scale}},
                 [&](const raft::handle_t& /*h*/) { unimplemented(); });
}

CUMLRKT_JOB(cuml_preprocessing_standard_scaler_fit)
CUMLRKT_JOB(cuml_preprocessing_standard_scaler_transform)
CUMLRKT_JOB(cuml_preprocessing_standard_scaler_inverse_transform)
}
