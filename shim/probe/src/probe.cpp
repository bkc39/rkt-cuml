#include "cumlrkt/probe.h"

#include <stdexcept>

#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

namespace {

int unimplemented() {
  return raftrkt::translate_exceptions(
      cumlrkt::last_error_slot(),
      [] { throw std::logic_error("unimplemented"); });
}

}  // namespace

extern "C" {

const char* cuml_probe_last_error(void) {
  return cumlrkt::last_error_slot().message.data();
}

int cuml_probe_last_error_kind(void) {
  return static_cast<int>(cumlrkt::last_error_slot().kind);
}

int cuml_probe_check_abi(const rr_abi_tag* loaded) {
  return cumlrkt::check_abi(loaded);
}

const char* cuml_probe_libcuml_version(void) {
  return cumlrkt::libcuml_release.data();
}

int cuml_probe_model_create(void* /*handle*/, rr_deposit* /*out*/,
                            int64_t /*payload_bytes*/,
                            int32_t /*region_length*/, int32_t /*kind*/) {
  return unimplemented();
}

void cuml_probe_model_create_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_create>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}

int cuml_probe_model_touch(void* /*handle*/, const rr_view* /*anchor*/,
                           int32_t /*gate*/, int64_t* /*payload_bytes*/) {
  return unimplemented();
}

void cuml_probe_model_touch_job(void* args, size_t args_len,
                                rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_touch>(args, args_len, outcome,
                                            cumlrkt::last_error_slot());
}

int cuml_probe_model_release(void* /*handle*/, const rr_view* /*anchor*/) {
  return unimplemented();
}

void cuml_probe_model_release_job(void* args, size_t args_len,
                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_release>(args, args_len, outcome,
                                              cumlrkt::last_error_slot());
}

int cuml_probe_model_counts(int64_t* /*created*/, int64_t* /*released*/,
                            int64_t* /*destroyed*/, int64_t* /*stale*/) {
  return unimplemented();
}

int cuml_probe_gate_create(int32_t* /*gate*/) {
  return unimplemented();
}

int cuml_probe_gate_open(int32_t /*gate*/) {
  return unimplemented();
}

int cuml_probe_gate_waiters(int32_t /*gate*/, int32_t* /*waiters*/) {
  return unimplemented();
}
}
