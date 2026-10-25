#lang racket/base

(require (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/native
                  cuml-family-libcuml-version
                  cuml-family-library
                  load-family)
         (only-in ffi/unsafe _double _int32 _int64)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in raft/unsafe define-job))

(provide cuml-metrics-adjusted-rand-score-job
         cuml-metrics-contingency-matrix-job
         cuml-metrics-silhouette-samples-job
         cuml-metrics-silhouette-score-job
         cuml-metrics-trustworthiness-job
         libcuml-version)

(define-runtime-path native-libs-dir "../native-libs")

(define metrics (load-family "metrics" native-libs-dir))
(define library (cuml-family-library metrics))
(define libcuml-version (cuml-family-libcuml-version metrics))

(define-job cuml-metrics-contingency-matrix-job
            library
            "cuml_metrics_contingency_matrix_job"
            (handle view view deposit)
            #:exn exn:fail:cuml)

(define-job cuml-metrics-adjusted-rand-score-job
            library
            "cuml_metrics_adjusted_rand_score_job"
            (handle view view (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-metrics-silhouette-score-job
            library
            "cuml_metrics_silhouette_score_job"
            (handle view view _int32 _int64 (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-metrics-silhouette-samples-job
            library
            "cuml_metrics_silhouette_samples_job"
            (handle view view deposit _int32 _int64 (out _double))
            #:exn exn:fail:cuml)

(define-job cuml-metrics-trustworthiness-job
            library
            "cuml_metrics_trustworthiness_job"
            (handle view view _int64 _int32 _int64 (out _double))
            #:exn exn:fail:cuml)
