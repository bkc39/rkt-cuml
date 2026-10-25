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

/* Allocates its outputs after every check and deposits them in out: region 0
   is the d float64 means, always; when with_std, region 1 the d float64
   variances and region 2 the d float64 scales. A column chunk of the float64
   scratch holds max(1, scratch_elements / n) columns. */
CUMLRKT_API int cuml_preprocessing_standard_scaler_fit(
    void* handle, const rr_view* x, rr_deposit* out, int32_t with_mean,
    int32_t with_std, int64_t scratch_elements, int64_t* n_samples);
CUMLRKT_API void cuml_preprocessing_standard_scaler_fit_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* Each transform takes mean and scale as d float64 vectors, or NULL to skip
   that step, and deposits in out region 0, the n x d row-major result in X's
   dtype. */
CUMLRKT_API int cuml_preprocessing_standard_scaler_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* out, int64_t n_features);
CUMLRKT_API void cuml_preprocessing_standard_scaler_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

CUMLRKT_API int cuml_preprocessing_standard_scaler_inverse_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* out, int64_t n_features);
CUMLRKT_API void cuml_preprocessing_standard_scaler_inverse_transform_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
