#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native cuml-family-library load-family)
         (only-in ffi/unsafe _double* _int32 _int64 _uint64)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide cuml-manifold-umap-fit-job
         cuml-manifold-umap-transform-job)

(define-runtime-path native-libs-dir "../native-libs")

(define library (cuml-family-library (load-family "manifold" native-libs-dir)))

(define-job cuml-manifold-umap-fit-job
            library
            "cuml_manifold_umap_fit_job"
            (handle view
                    view
                    deposit
                    _int64
                    _int64
                    _int32
                    _int64
                    _double*
                    _double*
                    _double*
                    _int32
                    _int64
                    _uint64
                    _int32
                    _int32
                    _int32)
            #:exn exn:fail:cuml)

(define-job cuml-manifold-umap-transform-job
            library
            "cuml_manifold_umap_transform_job"
            (handle view
                    view
                    view
                    deposit
                    _int64
                    _int64
                    _int32
                    _int64
                    _double*
                    _double*
                    _double*
                    _int64
                    _uint64
                    _int32
                    _int32)
            #:exn exn:fail:cuml)
