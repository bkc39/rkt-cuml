#ifndef CUMLRKT_LINEAR_MODEL_H
#define CUMLRKT_LINEAR_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
#include "raftrkt/job.h"

#define CUML_LINEAR_MODEL_OLS_SVD 0
#define CUML_LINEAR_MODEL_OLS_EIG 1
#define CUML_LINEAR_MODEL_OLS_QR 2
#define CUML_LINEAR_MODEL_OLS_SVD_QR 3

#ifdef __cplusplus
extern "C" {
#endif

CUMLRKT_API const char* cuml_linear_model_last_error(void);
CUMLRKT_API int cuml_linear_model_last_error_kind(void);
CUMLRKT_API int cuml_linear_model_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_linear_model_libcuml_version(void);

/* sample_weight may be NULL. Region 0 of out is the d coefficients in X's
   dtype; *intercept is exact in X's dtype, and 0 without fit_intercept. */
CUMLRKT_API int cuml_linear_model_linear_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, rr_deposit* out, int32_t fit_intercept,
    int32_t algorithm, double* intercept);
CUMLRKT_API void cuml_linear_model_linear_regression_fit_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* Region 0 of out is the n predictions X coef + intercept in X's dtype. */
CUMLRKT_API int cuml_linear_model_gemm_predict(void* handle, const rr_view* x,
                                               const rr_view* coef,
                                               double intercept,
                                               rr_deposit* out);
CUMLRKT_API void cuml_linear_model_gemm_predict_job(void* args, size_t args_len,
                                                    rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
