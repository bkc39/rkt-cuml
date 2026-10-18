#ifndef CUMLRKT_CLUSTER_H
#define CUMLRKT_CLUSTER_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/job.h"

#ifdef __cplusplus
extern "C" {
#endif

CUMLRKT_API const char* cuml_cluster_last_error(void);
CUMLRKT_API int cuml_cluster_last_error_kind(void);
CUMLRKT_API int cuml_cluster_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_cluster_libcuml_version(void);

CUMLRKT_API int cuml_cluster_make_blobs(void* handle, const rr_view* out,
                                        const rr_view* labels,
                                        int32_t n_clusters, double cluster_std,
                                        int32_t shuffle, double center_box_min,
                                        double center_box_max, uint64_t seed);
CUMLRKT_API void cuml_cluster_make_blobs_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

#define CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS 0
#define CUML_CLUSTER_KMEANS_INIT_RANDOM 1
#define CUML_CLUSTER_KMEANS_INIT_ARRAY 2

CUMLRKT_API int cuml_cluster_kmeans_fit(
    void* handle, const rr_view* x, const rr_view* sample_weight,
    const rr_view* init, const rr_view* centroids, const rr_view* labels,
    int32_t n_clusters, int32_t init_method, int32_t max_iter, double tol,
    int32_t n_init, double oversampling_factor, int32_t batch_samples,
    uint64_t seed, double* inertia, int32_t* n_iter);
CUMLRKT_API void cuml_cluster_kmeans_fit_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_kmeans_predict(
    void* handle, const rr_view* centroids, const rr_view* x,
    const rr_view* sample_weight, const rr_view* labels, int32_t batch_samples,
    double* inertia);
CUMLRKT_API void cuml_cluster_kmeans_predict_job(void* args, size_t args_len,
                                                 rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_kmeans_transform(void* handle,
                                              const rr_view* centroids,
                                              const rr_view* x,
                                              const rr_view* distances,
                                              int32_t batch_samples);
CUMLRKT_API void cuml_cluster_kmeans_transform_job(void* args, size_t args_len,
                                                   rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
