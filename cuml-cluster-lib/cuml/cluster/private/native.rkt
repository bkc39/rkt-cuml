#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-call
                  cuml-family-library
                  cuml-family-libcuml-version
                  load-family)
         (only-in ffi/unsafe _double _fun _int _int32 _pointer _uint64)
         (only-in ffi/unsafe/define define-ffi-definer)
         (only-in ffi/unsafe/define/conventions convention:hyphen->underscore)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide call/cluster
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
