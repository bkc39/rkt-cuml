#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-libcuml-version
                  cuml-family-library
                  load-family)
         (only-in ffi/unsafe _int32 _int64)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide cuml-preprocessing-standard-scaler-fit-job
         cuml-preprocessing-standard-scaler-inverse-transform-job
         cuml-preprocessing-standard-scaler-transform-job
         libcuml-version)

(define-runtime-path native-libs-dir "../native-libs")

(define preprocessing (load-family "preprocessing" native-libs-dir))
(define library (cuml-family-library preprocessing))
(define libcuml-version (cuml-family-libcuml-version preprocessing))

(define-job cuml-preprocessing-standard-scaler-fit-job
            library
            "cuml_preprocessing_standard_scaler_fit_job"
            (handle view deposit _int32 _int32 _int64 (out _int64))
            #:exn exn:fail:cuml)

(define-job cuml-preprocessing-standard-scaler-transform-job
            library
            "cuml_preprocessing_standard_scaler_transform_job"
            (handle view view view deposit _int64)
            #:exn exn:fail:cuml)

(define-job cuml-preprocessing-standard-scaler-inverse-transform-job
            library
            "cuml_preprocessing_standard_scaler_inverse_transform_job"
            (handle view view view deposit _int64)
            #:exn exn:fail:cuml)
