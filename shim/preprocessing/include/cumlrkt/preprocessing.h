#ifndef CUMLRKT_PREPROCESSING_H
#define CUMLRKT_PREPROCESSING_H

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

CUMLRKT_API const char* cuml_preprocessing_last_error(void);
CUMLRKT_API int cuml_preprocessing_last_error_kind(void);
CUMLRKT_API int cuml_preprocessing_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_preprocessing_libcuml_version(void);

/* Allocates after every check and deposits in out: region 0 the d float64
   means, always; when with_std, region 1 the d float64 population variances
   and region 2 the d float64 scales. Statistics accumulate in float64 over
   column chunks of at most max(1, scratch_elements / n) columns. */
CUMLRKT_API int cuml_preprocessing_standard_scaler_fit(
    void* handle, const rr_view* x, rr_deposit* out, int32_t with_mean,
    int32_t with_std, int64_t scratch_elements, int64_t* n_samples);
CUMLRKT_API void cuml_preprocessing_standard_scaler_fit_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* mean and scale are NULL or d float64 vectors. Region 0 of out is the
   n x d row-major result in X's dtype: (x - T(mean)) / T(scale), each step
   skipped when its vector is NULL, so both NULL is a copy. */
CUMLRKT_API int cuml_preprocessing_standard_scaler_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* out, int64_t n_features);
CUMLRKT_API void cuml_preprocessing_standard_scaler_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* t * T(scale) + T(mean), unfused, each step skipped when NULL. */
CUMLRKT_API int cuml_preprocessing_standard_scaler_inverse_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* out, int64_t n_features);
CUMLRKT_API void cuml_preprocessing_standard_scaler_inverse_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
