#include <cstddef>
#include <cstdint>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

namespace {

int unimplemented() {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw cumlrkt::logic_error("unimplemented");
  });
}

}  // namespace

extern "C" {

int cuml_cluster_hdbscan_fit(void* /*handle*/, const rr_view* /*x*/,
                             rr_deposit* /*model_out*/, rr_deposit* /*out*/,
                             int64_t /*min_cluster_size*/,
                             int64_t /*min_samples*/,
                             double /*cluster_selection_epsilon*/,
                             int64_t /*max_cluster_size*/, double /*alpha*/,
                             int32_t /*cluster_selection_method*/,
                             int32_t /*allow_single_cluster*/,
                             int64_t* /*opaque_bytes*/) {
  return unimplemented();
}

void cuml_cluster_hdbscan_fit_job(void* args, size_t args_len,
                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_fit>(args, args_len, outcome,
                                              cumlrkt::last_error_slot());
}

int cuml_cluster_hdbscan_approximate_predict(void* /*handle*/,
                                             const rr_view* /*model*/,
                                             const rr_view* /*x*/,
                                             const rr_view* /*points*/,
                                             rr_deposit* /*out*/) {
  return unimplemented();
}

void cuml_cluster_hdbscan_approximate_predict_job(void* args, size_t args_len,
                                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_approximate_predict>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_hdbscan_membership_vector(void* /*handle*/,
                                           const rr_view* /*model*/,
                                           const rr_view* /*x*/,
                                           const rr_view* /*points*/,
                                           rr_deposit* /*out*/,
                                           int64_t /*batch_size*/) {
  return unimplemented();
}

void cuml_cluster_hdbscan_membership_vector_job(void* args, size_t args_len,
                                                rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_membership_vector>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_hdbscan_all_points_membership_vectors(
    void* /*handle*/, const rr_view* /*model*/, const rr_view* /*x*/,
    rr_deposit* /*out*/, int64_t /*batch_size*/) {
  return unimplemented();
}

void cuml_cluster_hdbscan_all_points_membership_vectors_job(
    void* args, size_t args_len, rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_all_points_membership_vectors>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_model_release(void* /*handle*/, const rr_view* /*model*/) {
  return unimplemented();
}

void cuml_cluster_model_release_job(void* args, size_t args_len,
                                    rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_model_release>(args, args_len, outcome,
                                                cumlrkt::last_error_slot());
}
}
