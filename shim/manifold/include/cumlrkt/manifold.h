#ifndef CUMLRKT_MANIFOLD_H
#define CUMLRKT_MANIFOLD_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
#include "raftrkt/job.h"

#define CUML_MANIFOLD_METRIC_EUCLIDEAN 0
#define CUML_MANIFOLD_METRIC_SQEUCLIDEAN 1
#define CUML_MANIFOLD_METRIC_MANHATTAN 2
#define CUML_MANIFOLD_METRIC_CHEBYSHEV 3
#define CUML_MANIFOLD_METRIC_COSINE 4
#define CUML_MANIFOLD_INIT_RANDOM 0
#define CUML_MANIFOLD_INIT_SPECTRAL 1
#define CUML_MANIFOLD_BUILD_BRUTE_FORCE 0
#define CUML_MANIFOLD_BUILD_NN_DESCENT 1

#ifdef __cplusplus
extern "C" {
#endif

CUMLRKT_API const char* cuml_manifold_last_error(void);
CUMLRKT_API int cuml_manifold_last_error_kind(void);
CUMLRKT_API int cuml_manifold_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_manifold_libcuml_version(void);

/* y may be NULL. Region 0 of out is the n x n_components row-major float32
   embedding. */
CUMLRKT_API int cuml_manifold_umap_fit(
    void* handle, const rr_view* x, const rr_view* y, rr_deposit* out,
    int64_t n_neighbors, int64_t n_components, int32_t epochs_given,
    int64_t n_epochs, double learning_rate, double a,
    double b, int32_t init, int64_t negative_sample_rate,
    uint64_t random_state, int32_t deterministic, int32_t build_algo,
    int32_t metric);
CUMLRKT_API void cuml_manifold_umap_fit_job(void* args, size_t args_len,
                                            rr_job_outcome* outcome);

/* Region 0 of out is the m x n_components row-major float32 projection. */
CUMLRKT_API int cuml_manifold_umap_transform(
    void* handle, const rr_view* x, const rr_view* train_x,
    const rr_view* embedding, rr_deposit* out, int64_t n_neighbors,
    int64_t n_components, int32_t epochs_given, int64_t n_epochs,
    double learning_rate, double a, double b,
    int64_t negative_sample_rate, uint64_t random_state, int32_t deterministic,
    int32_t metric);
CUMLRKT_API void cuml_manifold_umap_transform_job(void* args, size_t args_len,
                                                  rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
