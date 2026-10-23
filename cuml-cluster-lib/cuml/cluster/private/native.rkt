#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-call
                  cuml-family-library
                  cuml-family-libcuml-version
                  load-family)
         (only-in ffi/unsafe _double _double* _fun _int _int32 _pointer _uint64)
         (only-in ffi/unsafe/define define-ffi-definer)
         (only-in ffi/unsafe/define/conventions convention:hyphen->underscore)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide call/cluster
         cuml-cluster-kmeans-fit-job
         cuml-cluster-kmeans-predict-job
         cuml-cluster-kmeans-score-job
         cuml-cluster-kmeans-transform-job
         cuml-cluster-make-blobs
         cuml-cluster-make-blobs-job
         libcuml-version)

(define-runtime-path native-libs-dir "../native-libs")

(define cluster (load-family "cluster" native-libs-dir))
(define library (cuml-family-library cluster))
(define call/cluster (cuml-family-call cluster))
(define libcuml-version (cuml-family-libcuml-version cluster))

(define-ffi-definer define-cluster library #:make-c-id convention:hyphen->underscore)

(define-cluster cuml-cluster-make-blobs
                (_fun _pointer _pointer _pointer _int32 _double _int32 _double _double _uint64
                      -> _int))

(define-job cuml-cluster-make-blobs-job
            library
            "cuml_cluster_make_blobs_job"
            (handle view view _int32 _double _int32 _double _double _uint64)
            #:exn exn:fail:cuml)

(define-job cuml-cluster-kmeans-fit-job
            library
            "cuml_cluster_kmeans_fit_job"
            (handle view
                    view
                    view
                    deposit
                    _int32
                    _int32
                    _int32
                    _double*
                    _int32
                    _double*
                    _int32
                    _uint64
                    (out _double)
                    (out _int32))
            #:exn exn:fail:cuml)

(define-job cuml-cluster-kmeans-predict-job
            library
            "cuml_cluster_kmeans_predict_job"
            (handle view view deposit _int32)
            #:exn exn:fail:cuml)

(define-job cuml-cluster-kmeans-score-job
            library
            "cuml_cluster_kmeans_score_job"
            (handle view view view _int32 (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-cluster-kmeans-transform-job
            library
            "cuml_cluster_kmeans_transform_job"
            (handle view view deposit _int32)
            #:exn exn:fail:cuml)
