#ifndef CUMLRKT_METRICS_H
#define CUMLRKT_METRICS_H

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

#define CUML_METRICS_METRIC_EUCLIDEAN 0
#define CUML_METRICS_METRIC_SQEUCLIDEAN 1
#define CUML_METRICS_METRIC_MANHATTAN 2
#define CUML_METRICS_METRIC_COSINE 3
#define CUML_METRICS_METRIC_CHEBYSHEV 4

CUMLRKT_API const char* cuml_metrics_last_error(void);
CUMLRKT_API int cuml_metrics_last_error_kind(void);
CUMLRKT_API int cuml_metrics_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_metrics_libcuml_version(void);

/* Deposits in out: region 0 the k1 x k2 int64 counts, row-major; region 1
   the k1 distinct values of labels_true, ascending, in its dtype; region 2
   the k2 distinct values of labels_pred, likewise. */
CUMLRKT_API int cuml_metrics_contingency_matrix(void* handle,
                                                const rr_view* labels_true,
                                                const rr_view* labels_pred,
                                                rr_deposit* out);
CUMLRKT_API void cuml_metrics_contingency_matrix_job(void* args,
                                                     size_t args_len,
                                                     rr_job_outcome* outcome);

CUMLRKT_API int cuml_metrics_adjusted_rand_score(void* handle,
                                                 const rr_view* labels_true,
                                                 const rr_view* labels_pred,
                                                 double* score);
CUMLRKT_API void cuml_metrics_adjusted_rand_score_job(void* args,
                                                      size_t args_len,
                                                      rr_job_outcome* outcome);

CUMLRKT_API int cuml_metrics_silhouette_score(void* handle, const rr_view* x,
                                              const rr_view* labels,
                                              int32_t metric,
                                              int64_t chunk_size,
                                              double* score);
CUMLRKT_API void cuml_metrics_silhouette_score_job(void* args, size_t args_len,
                                                   rr_job_outcome* outcome);

/* Deposits in out: region 0 the n per-sample scores, in X's dtype. */
CUMLRKT_API int cuml_metrics_silhouette_samples(void* handle, const rr_view* x,
                                                const rr_view* labels,
                                                rr_deposit* out, int32_t metric,
                                                int64_t chunk_size,
                                                double* score);
CUMLRKT_API void cuml_metrics_silhouette_samples_job(void* args,
                                                     size_t args_len,
                                                     rr_job_outcome* outcome);

CUMLRKT_API int cuml_metrics_trustworthiness(void* handle, const rr_view* x,
                                             const rr_view* x_embedded,
                                             int64_t n_neighbors,
                                             int32_t metric, int64_t batch_size,
                                             double* score);
CUMLRKT_API void cuml_metrics_trustworthiness_job(void* args, size_t args_len,
                                                  rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
