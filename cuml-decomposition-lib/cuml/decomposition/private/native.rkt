#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-libcuml-version
                  cuml-family-library
                  load-family)
         (only-in ffi/unsafe _double _double* _int32 _int64)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide cuml-decomposition-pca-fit-job
         cuml-decomposition-pca-inverse-transform-job
         cuml-decomposition-pca-transform-job
         cuml-decomposition-tsvd-fit-job
         cuml-decomposition-tsvd-inverse-transform-job
         cuml-decomposition-tsvd-transform-job
         libcuml-version)

(define-runtime-path native-libs-dir "../native-libs")

(define decomposition (load-family "decomposition" native-libs-dir))
(define library (cuml-family-library decomposition))
(define libcuml-version (cuml-family-libcuml-version decomposition))

(define-job cuml-decomposition-pca-fit-job
            library
            "cuml_decomposition_pca_fit_job"
            (handle view deposit _int32 _int64 _int32 _double* _int64 (out _double) (out _int64))
            #:exn exn:fail:cuml)

(define-job cuml-decomposition-pca-transform-job
            library
            "cuml_decomposition_pca_transform_job"
            (handle view view view view deposit _int32)
            #:exn exn:fail:cuml)

(define-job cuml-decomposition-pca-inverse-transform-job
            library
            "cuml_decomposition_pca_inverse_transform_job"
            (handle view view view view deposit _int32)
            #:exn exn:fail:cuml)

(define-job cuml-decomposition-tsvd-fit-job
            library
            "cuml_decomposition_tsvd_fit_job"
            (handle view deposit _int64 _int32 _double* _int64 _int32)
            #:exn exn:fail:cuml)

(define-job cuml-decomposition-tsvd-transform-job
            library
            "cuml_decomposition_tsvd_transform_job"
            (handle view view deposit)
            #:exn exn:fail:cuml)

(define-job cuml-decomposition-tsvd-inverse-transform-job
            library
            "cuml_decomposition_tsvd_inverse_transform_job"
            (handle view view deposit)
            #:exn exn:fail:cuml)
