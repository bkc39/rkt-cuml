#include <cstdint>
#include <stdexcept>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

extern "C" {

int cuml_cluster_kmeans_fit(void* /*handle*/, const rr_view* /*x*/,
                            const rr_view* /*sample_weight*/,
                            const rr_view* /*init*/,
                            const rr_view* /*centroids*/,
                            const rr_view* /*labels*/, int32_t /*n_clusters*/,
                            int32_t /*init_method*/, int32_t /*max_iter*/,
                            double /*tol*/, int32_t /*n_init*/,
                            double /*oversampling_factor*/,
                            int32_t /*batch_samples*/, uint64_t /*seed*/,
                            double* /*inertia*/, int32_t* /*n_iter*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

int cuml_cluster_kmeans_predict(void* /*handle*/, const rr_view* /*centroids*/,
                                const rr_view* /*x*/,
                                const rr_view* /*sample_weight*/,
                                const rr_view* /*labels*/,
                                int32_t /*batch_samples*/,
                                double* /*inertia*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

int cuml_cluster_kmeans_transform(void* /*handle*/,
                                  const rr_view* /*centroids*/,
                                  const rr_view* /*x*/,
                                  const rr_view* /*distances*/,
                                  int32_t /*batch_samples*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

void cuml_cluster_kmeans_fit_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_fit>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}

void cuml_cluster_kmeans_predict_job(void* args, size_t args_len,
                                     rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_predict>(args, args_len, outcome,
                                                 cumlrkt::last_error_slot());
}

void cuml_cluster_kmeans_transform_job(void* args, size_t args_len,
                                       rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_transform>(args, args_len, outcome,
                                                   cumlrkt::last_error_slot());
}
}
