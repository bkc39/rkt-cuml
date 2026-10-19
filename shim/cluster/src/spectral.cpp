#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

extern "C" {

int cuml_cluster_spectral_clustering_fit(
    void* /*handle*/, const rr_view* /*x*/, rr_deposit* /*out*/,
    int64_t /*n_clusters*/, int64_t /*n_components*/,
    int32_t /*auto_components*/, int64_t /*n_neighbors*/, int32_t /*n_init*/,
    double /*eigen_tol*/, uint64_t /*seed*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

void cuml_cluster_spectral_clustering_fit_job(void* args, size_t args_len,
                                              rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_spectral_clustering_fit>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_spectral_clustering_fit_graph(
    void* /*handle*/, const rr_view* /*rows*/, const rr_view* /*cols*/,
    const rr_view* /*vals*/, int64_t /*n_samples*/, rr_deposit* /*out*/,
    int64_t /*n_clusters*/, int64_t /*n_components*/,
    int32_t /*auto_components*/, int32_t /*n_init*/, double /*eigen_tol*/,
    uint64_t /*seed*/) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [] {
    throw std::logic_error("unimplemented");
  });
}

void cuml_cluster_spectral_clustering_fit_graph_job(void* args, size_t args_len,
                                                    rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_spectral_clustering_fit_graph>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}
}
