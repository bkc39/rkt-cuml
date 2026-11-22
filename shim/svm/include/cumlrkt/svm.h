#ifndef CUMLRKT_SVM_H
#define CUMLRKT_SVM_H

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

#define CUML_SVM_KERNEL_LINEAR 0
#define CUML_SVM_KERNEL_POLY 1
#define CUML_SVM_KERNEL_RBF 2
#define CUML_SVM_KERNEL_SIGMOID 3

#define CUML_SVM_GAMMA_NUMBER 0
#define CUML_SVM_GAMMA_SCALE 1
#define CUML_SVM_GAMMA_AUTO 2

#define CUML_SVM_APPLY_DECISION 0
#define CUML_SVM_APPLY_LABELS 1

CUMLRKT_API const char* cuml_svm_last_error(void);
CUMLRKT_API int cuml_svm_last_error_kind(void);
CUMLRKT_API int cuml_svm_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_svm_libcuml_version(void);

/* codes: int32 0/1 per row of x, 1 for the second class. sample_weight: NULL
   or one per row in x's dtype. Deposits in out: region 0 the n_s x d support
   vectors, row-major; region 1 their int32 row indices, ascending; region 2
   the 1 x n_s dual coefficients y*alpha; region 3 the intercept.
   empty_class: -1, or the code of a class whose rows all have weight 0, in
   which case nothing is fitted or deposited. */
CUMLRKT_API int cuml_svm_svc_fit(void* handle, const rr_view* x,
                                 const rr_view* codes,
                                 const rr_view* sample_weight, rr_deposit* out,
                                 int32_t kernel, int32_t degree,
                                 int32_t gamma_mode, double gamma, double coef0,
                                 double C, double tol, double cache_size,
                                 int32_t max_iter, int32_t nochange_steps,
                                 double* gamma_used, int32_t* n_iter,
                                 int32_t* n_negative, int32_t* n_positive,
                                 int32_t* empty_class);
CUMLRKT_API void cuml_svm_svc_fit_job(void* args, size_t args_len,
                                      rr_job_outcome* outcome);

/* Deposits in out one value per row of x: the decision function in x's
   dtype (mode 0), or classes[decision >= 0] in classes' dtype (mode 1;
   int32 0/1 when classes is NULL). */
CUMLRKT_API int cuml_svm_svc_apply(void* handle, const rr_view* x,
                                   const rr_view* support_vectors,
                                   const rr_view* dual_coef,
                                   const rr_view* intercept,
                                   const rr_view* classes, rr_deposit* out,
                                   int32_t kernel, int32_t degree, double gamma,
                                   double coef0, double cache_size,
                                   int32_t mode);
CUMLRKT_API void cuml_svm_svc_apply_job(void* args, size_t args_len,
                                        rr_job_outcome* outcome);

CUMLRKT_API int cuml_svm_label_classes(void* handle, const rr_view* y,
                                       rr_deposit* out);
CUMLRKT_API void cuml_svm_label_classes_job(void* args, size_t args_len,
                                            rr_job_outcome* outcome);
CUMLRKT_API int cuml_svm_label_encode(void* handle, const rr_view* y,
                                      const rr_view* classes, rr_deposit* out);
CUMLRKT_API void cuml_svm_label_encode_job(void* args, size_t args_len,
                                           rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
