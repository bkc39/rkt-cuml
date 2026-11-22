#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/labels label-jobs)
         (only-in cuml/core/private/native
                  cuml-family-libcuml-version
                  cuml-family-library
                  load-family)
         (only-in ffi/unsafe _double _int32)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide current-label-classes-job
         cuml-svm-label-classes-job
         cuml-svm-label-encode-job
         cuml-svm-svc-apply-job
         cuml-svm-svc-fit-job
         libcuml-version
         svm-label-jobs)

(define-runtime-path native-libs-dir "../native-libs")

(define svm (load-family "svm" native-libs-dir))
(define library (cuml-family-library svm))
(define libcuml-version (cuml-family-libcuml-version svm))

(define-job cuml-svm-svc-fit-job
            library
            "cuml_svm_svc_fit_job"
            (handle view
                    view
                    view
                    deposit
                    _int32
                    _int32
                    _int32
                    _double
                    _double
                    _double
                    _double
                    _double
                    _int32
                    _int32
                    (out _double)
                    (out _int32)
                    (out _int32)
                    (out _int32)
                    (out _int32))
            #:exn exn:fail:cuml)

(define-job cuml-svm-svc-apply-job
            library
            "cuml_svm_svc_apply_job"
            (handle view view view view view deposit _int32 _int32 _double _double _double _int32)
            #:exn exn:fail:cuml)

(define-job cuml-svm-label-classes-job
            library
            "cuml_svm_label_classes_job"
            (handle view deposit)
            #:exn exn:fail:cuml)

(define-job cuml-svm-label-encode-job
            library
            "cuml_svm_label_encode_job"
            (handle view view deposit)
            #:exn exn:fail:cuml)

(define current-label-classes-job
  (make-parameter cuml-svm-label-classes-job #f 'current-label-classes-job))

(define svm-label-jobs
  (label-jobs (lambda args (apply (current-label-classes-job) args)) cuml-svm-label-encode-job))
