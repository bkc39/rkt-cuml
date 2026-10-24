#ifndef CUMLRKT_CLUSTER_H
#define CUMLRKT_CLUSTER_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
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

/* Each entry allocates its outputs after every check and deposits them in
   out: fit's region 0 is the k x d centroids in X's dtype and region 1 the n
   int32 labels; predict's region 0 is the n int32 labels; transform's region
   0 is the n x k distances in the centroids' dtype. */
CUMLRKT_API int cuml_cluster_kmeans_fit(
    void* handle, const rr_view* x, const rr_view* sample_weight,
    const rr_view* init, rr_deposit* out, int32_t n_clusters,
    int32_t init_method, int32_t max_iter, double tol, int32_t n_init,
    double oversampling_factor, int32_t batch_samples, uint64_t seed,
    double* inertia, int32_t* n_iter);
CUMLRKT_API void cuml_cluster_kmeans_fit_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_kmeans_predict(void* handle,
                                            const rr_view* centroids,
                                            const rr_view* x, rr_deposit* out,
                                            int32_t batch_samples);
CUMLRKT_API void cuml_cluster_kmeans_predict_job(void* args, size_t args_len,
                                                 rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_kmeans_score(
    void* handle, const rr_view* centroids, const rr_view* x,
    const rr_view* sample_weight, int32_t batch_samples, double* inertia);
CUMLRKT_API void cuml_cluster_kmeans_score_job(void* args, size_t args_len,
                                               rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_kmeans_transform(void* handle,
                                              const rr_view* centroids,
                                              const rr_view* x, rr_deposit* out,
                                              int32_t batch_samples);
CUMLRKT_API void cuml_cluster_kmeans_transform_job(void* args, size_t args_len,
                                                   rr_job_outcome* outcome);

#define CUML_CLUSTER_DBSCAN_METRIC_EUCLIDEAN 0
#define CUML_CLUSTER_DBSCAN_METRIC_COSINE 1
#define CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED 2

/* Allocates its outputs after every check and deposits them in out: region 0
   is the n labels, region 1 the core-sample indices, ascending, both in
   out_dtype (RR_DTYPE_INT32 or RR_DTYPE_INT64). max_mbytes_per_batch 0 is
   cuML's automatic size. Under the cosine metric cuML runs on a copy of X. */
CUMLRKT_API int cuml_cluster_dbscan_fit(void* handle, const rr_view* x,
                                        const rr_view* sample_weight,
                                        rr_deposit* out, double eps,
                                        int32_t min_samples, int32_t metric,
                                        int32_t out_dtype,
                                        int64_t max_mbytes_per_batch);
CUMLRKT_API void cuml_cluster_dbscan_fit_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

#define CUML_CLUSTER_LINKAGE_METRIC_EUCLIDEAN 0
#define CUML_CLUSTER_LINKAGE_CONNECTIVITY_PAIRWISE 0
#define CUML_CLUSTER_LINKAGE_CONNECTIVITY_KNN 1

/* Allocates its outputs after every check and deposits them in out: region 0
   is the n int32 labels, region 1 the (n-1) x 2 int32 children, each row
   ascending. c is used only by the knn connectivity, and a c below
   -floor(log2 n) counts as -floor(log2 n). */
CUMLRKT_API int cuml_cluster_single_linkage_fit(
    void* handle, const rr_view* x, rr_deposit* out, int64_t n_clusters,
    int32_t metric, int32_t connectivity, int32_t c);
CUMLRKT_API void cuml_cluster_single_linkage_fit_job(void* args,
                                                     size_t args_len,
                                                     rr_job_outcome* outcome);

/* Each entry allocates its labels after every check and deposits them in out
   as region 0, n int32 labels. auto_components nonzero takes
   min(n_clusters, floor((n-1)/3)) components and ignores n_components;
   n_neighbors above n-1 counts as n-1. The graph is rows, cols and vals, one
   entry each, over n_samples samples. */
CUMLRKT_API int cuml_cluster_spectral_clustering_fit(
    void* handle, const rr_view* x, rr_deposit* out, int64_t n_clusters,
    int64_t n_components, int32_t auto_components, int64_t n_neighbors,
    int32_t n_init, double eigen_tol, uint64_t seed);
CUMLRKT_API void cuml_cluster_spectral_clustering_fit_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_spectral_clustering_fit_graph(
    void* handle, const rr_view* rows, const rr_view* cols, const rr_view* vals,
    int64_t n_samples, rr_deposit* out, int64_t n_clusters,
    int64_t n_components, int32_t auto_components, int32_t n_init,
    double eigen_tol, uint64_t seed);
CUMLRKT_API void cuml_cluster_spectral_clustering_fit_graph_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

#define CUML_CLUSTER_MODEL_KIND_HDBSCAN 1
#define CUML_CLUSTER_HDBSCAN_EOM 0
#define CUML_CLUSTER_HDBSCAN_LEAF 1

/* model_out regions: anchor, labels, tree (4), stabilities, selected. */
CUMLRKT_API int cuml_cluster_hdbscan_fit(
    void* handle, const rr_view* x, rr_deposit* model_out, rr_deposit* out,
    int64_t min_cluster_size, int64_t min_samples,
    double cluster_selection_epsilon, int64_t max_cluster_size, double alpha,
    int32_t cluster_selection_method, int32_t allow_single_cluster,
    int64_t* opaque_bytes);
CUMLRKT_API void cuml_cluster_hdbscan_fit_job(void* args, size_t args_len,
                                              rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_hdbscan_approximate_predict(void* handle,
                                                         const rr_view* model,
                                                         const rr_view* x,
                                                         const rr_view* points,
                                                         rr_deposit* out);
CUMLRKT_API void cuml_cluster_hdbscan_approximate_predict_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_hdbscan_membership_vector(
    void* handle, const rr_view* model, const rr_view* x, const rr_view* points,
    rr_deposit* out, int64_t batch_size);
CUMLRKT_API void cuml_cluster_hdbscan_membership_vector_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_hdbscan_all_points_membership_vectors(
    void* handle, const rr_view* model, const rr_view* x, rr_deposit* out,
    int64_t batch_size);
CUMLRKT_API void cuml_cluster_hdbscan_all_points_membership_vectors_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

CUMLRKT_API int cuml_cluster_model_release(void* handle, const rr_view* model);
CUMLRKT_API void cuml_cluster_model_release_job(void* args, size_t args_len,
                                                rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
