#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-libcuml-version
                  cuml-family-library
                  load-family)
         (only-in ffi/unsafe _double _int32)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide cuml-linear-model-elastic-net-fit-job
         cuml-linear-model-gemm-predict-job
         cuml-linear-model-linear-regression-fit-job
         cuml-linear-model-ridge-fit-job
         libcuml-version)

(define-runtime-path native-libs-dir "../native-libs")

(define linear-model (load-family "linear-model" native-libs-dir))
(define library (cuml-family-library linear-model))
(define libcuml-version (cuml-family-libcuml-version linear-model))

(define-job cuml-linear-model-linear-regression-fit-job
            library
            "cuml_linear_model_linear_regression_fit_job"
            (handle view view view deposit _int32 _int32 (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-linear-model-ridge-fit-job
            library
            "cuml_linear_model_ridge_fit_job"
            (handle view view view deposit _double _int32 _int32 (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-linear-model-elastic-net-fit-job
            library
            "cuml_linear_model_elastic_net_fit_job"
            (handle view
                    view
                    view
                    deposit
                    _double
                    _double
                    _int32
                    _int32
                    _double
                    _int32
                    (out _double)
                    (out _int32))
            #:exn exn:fail:cuml)

(define-job cuml-linear-model-gemm-predict-job
            library
            "cuml_linear_model_gemm_predict_job"
            (handle view view _double deposit)
            #:exn exn:fail:cuml)
