#ifndef CUMLRKT_DECOMPOSITION_H
#define CUMLRKT_DECOMPOSITION_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
#include "raftrkt/job.h"

#define CUML_DECOMPOSITION_SOLVER_FULL 0
#define CUML_DECOMPOSITION_SOLVER_JACOBI 1

#ifdef __cplusplus
extern "C" {
#endif

CUMLRKT_API const char* cuml_decomposition_last_error(void);
CUMLRKT_API int cuml_decomposition_last_error_kind(void);
CUMLRKT_API int cuml_decomposition_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_decomposition_libcuml_version(void);

/* k is n_components, or min(n, d) when all_components. Deposits in out:
   region 0 the k x d row-major components, region 1 the k explained
   variances, region 2 their ratios, region 3 the k singular values, region
   4 the d means, all in X's dtype. */
CUMLRKT_API int cuml_decomposition_pca_fit(
    void* handle, const rr_view* x, rr_deposit* out, int32_t all_components,
    int64_t n_components, int32_t solver, double tol, int64_t iterations,
    double* noise_variance, int64_t* n_samples);
CUMLRKT_API void cuml_decomposition_pca_fit_job(void* args, size_t args_len,
                                                rr_job_outcome* outcome);

/* explained_variance is read only when whiten, and may be NULL otherwise.
   Region 0 of out is the n x k row-major projection in X's dtype. */
CUMLRKT_API int cuml_decomposition_pca_transform(
    void* handle, const rr_view* x, const rr_view* components,
    const rr_view* mean, const rr_view* explained_variance, rr_deposit* out,
    int32_t whiten);
CUMLRKT_API void cuml_decomposition_pca_transform_job(void* args,
                                                      size_t args_len,
                                                      rr_job_outcome* outcome);

/* Region 0 of out is the n x d row-major reconstruction. */
CUMLRKT_API int cuml_decomposition_pca_inverse_transform(
    void* handle, const rr_view* x, const rr_view* components,
    const rr_view* mean, const rr_view* explained_variance, rr_deposit* out,
    int32_t whiten);
CUMLRKT_API void cuml_decomposition_pca_inverse_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* Deposits in out, all in X's dtype: region 0 the k x d row-major
   components, region 1 the k singular values, region 2 the k explained
   variances, region 3 their ratios, and when with_transform region 4 the
   n x k row-major transform of X. */
CUMLRKT_API int cuml_decomposition_tsvd_fit(void* handle, const rr_view* x,
                                            rr_deposit* out,
                                            int64_t n_components,
                                            int32_t solver, double tol,
                                            int64_t iterations,
                                            int32_t with_transform);
CUMLRKT_API void cuml_decomposition_tsvd_fit_job(void* args, size_t args_len,
                                                 rr_job_outcome* outcome);

/* Region 0 of out is x times the transposed components, n x k, row-major;
   x and components share a dtype. */
CUMLRKT_API int cuml_decomposition_tsvd_transform(void* handle,
                                                  const rr_view* x,
                                                  const rr_view* components,
                                                  rr_deposit* out);
CUMLRKT_API void cuml_decomposition_tsvd_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* Region 0 of out is x times the components, n x d, row-major. */
CUMLRKT_API int cuml_decomposition_tsvd_inverse_transform(
    void* handle, const rr_view* x, const rr_view* components,
    rr_deposit* out);
CUMLRKT_API void cuml_decomposition_tsvd_inverse_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
