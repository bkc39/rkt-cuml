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

#define CUML_LINEAR_MODEL_RIDGE_SVD 0
#define CUML_LINEAR_MODEL_RIDGE_EIG 1

#define CUML_LINEAR_MODEL_CD_CYCLIC 0
#define CUML_LINEAR_MODEL_CD_RANDOM 1

#define CUML_LINEAR_MODEL_PENALTY_NONE 0
#define CUML_LINEAR_MODEL_PENALTY_L1 1
#define CUML_LINEAR_MODEL_PENALTY_L2 2
#define CUML_LINEAR_MODEL_PENALTY_ELASTICNET 3

#define CUML_LINEAR_MODEL_CLASS_WEIGHT_NONE 0
#define CUML_LINEAR_MODEL_CLASS_WEIGHT_BALANCED 1
#define CUML_LINEAR_MODEL_CLASS_WEIGHT_GIVEN 2

#define CUML_LINEAR_MODEL_APPLY_DECISION 0
#define CUML_LINEAR_MODEL_APPLY_LABELS 1
#define CUML_LINEAR_MODEL_APPLY_PROBA 2
#define CUML_LINEAR_MODEL_APPLY_LOG_PROBA 3

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

/* sample_weight may be NULL. alpha is the penalty in X's units. Region 0 of out
   is the d coefficients in X's dtype; *intercept is exact in X's dtype, and 0
   without fit_intercept. */
CUMLRKT_API int cuml_linear_model_ridge_fit(void* handle, const rr_view* x,
                                            const rr_view* y,
                                            const rr_view* sample_weight,
                                            rr_deposit* out, double alpha,
                                            int32_t fit_intercept,
                                            int32_t solver, double* intercept);
CUMLRKT_API void cuml_linear_model_ridge_fit_job(void* args, size_t args_len,
                                                 rr_job_outcome* outcome);

/* sample_weight may be NULL. alpha and l1_ratio are Python cuML's ElasticNet
   parameters in X's and y's units. Region 0 of out is the d coefficients in X's
   dtype; *intercept is exact in X's dtype, and 0 without fit_intercept; *n_iter
   is the number of CD epochs, 0 when nothing was solved. */
CUMLRKT_API int cuml_linear_model_elastic_net_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, rr_deposit* out, double alpha,
    double l1_ratio, int32_t fit_intercept, int32_t max_iter, double tol,
    int32_t selection, double* intercept, int32_t* n_iter);
CUMLRKT_API void cuml_linear_model_elastic_net_fit_job(void* args,
                                                       size_t args_len,
                                                       rr_job_outcome* outcome);

/* Region 0 of out is the n predictions X coef + intercept in X's dtype. */
CUMLRKT_API int cuml_linear_model_gemm_predict(void* handle, const rr_view* x,
                                               const rr_view* coef,
                                               double intercept,
                                               rr_deposit* out);
CUMLRKT_API void cuml_linear_model_gemm_predict_job(void* args, size_t args_len,
                                                    rr_job_outcome* outcome);

/* y holds class labels (int32, int64, or integral float32/float64);
   sample_weight and class_weight may be NULL; class_weight (class_weight_mode
   2) is a float64 vector of one weight per class, in sorted class order.
   Region 0 of out is coef (k' x d, X's dtype, row-major; k' = 1 for two
   classes), region 1 the k' intercepts, region 2 the k sorted classes in y's
   dtype. */
CUMLRKT_API int cuml_linear_model_logistic_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, const rr_view* class_weight,
    rr_deposit* out, int32_t penalty, double C,
    double l1_ratio, double tol, int32_t max_iter, int32_t linesearch_max_iter,
    int32_t lbfgs_memory, int32_t fit_intercept, int32_t penalty_normalized,
    int32_t class_weight_mode, int32_t* n_iter);
CUMLRKT_API void cuml_linear_model_logistic_regression_fit_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* mode 0: region 0 is the n x k' decision; 1: the n labels, classes' dtype,
   or int32 codes when classes is NULL; 2 and 3: the n x k probabilities and
   their logarithms, X's dtype. */
CUMLRKT_API int cuml_linear_model_logistic_regression_apply(
    void* handle, const rr_view* x, const rr_view* coef,
    const rr_view* intercept, const rr_view* classes, int32_t mode,
    rr_deposit* out);
CUMLRKT_API void cuml_linear_model_logistic_regression_apply_job(
    void* args, size_t args_len, rr_job_outcome* outcome);

/* Region 0 of out is y's int32 codes in the sorted classes, -1 when absent. */
CUMLRKT_API int cuml_linear_model_label_encode(void* handle, const rr_view* y,
                                               const rr_view* classes,
                                               rr_deposit* out);
CUMLRKT_API void cuml_linear_model_label_encode_job(void* args, size_t args_len,
                                                    rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
