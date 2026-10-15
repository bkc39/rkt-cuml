#lang racket/base

(require (only-in raft current-device-resources device-matrix device-vector)
         (only-in raft/unsafe with-array-views)
         (only-in "native.rkt" call/cluster cuml-cluster-make-blobs cuml-cluster-make-blobs-job))

(provide blobs)

(define (blobs n-samples
               n-features
               #:centers [centers 3]
               #:cluster-std [cluster-std 1.0]
               #:seed [seed 0]
               #:dtype [element-type 'float32]
               #:job? [job? #t]
               #:resources [resources (current-device-resources)])
  (define X (device-matrix n-samples n-features #:dtype element-type #:resources resources))
  (define labels (device-vector n-samples #:dtype 'int32 #:resources resources))
  (define std (exact->inexact cluster-std))
  (cond
    [job? (cuml-cluster-make-blobs-job 'blobs resources X labels centers std 1 -10.0 10.0 seed)]
    [else
     (with-array-views #:resources [handle resources] ([x X] [y labels])
       (call/cluster 'blobs
                     (lambda ()
                       (cuml-cluster-make-blobs handle x y centers std 1 -10.0 10.0 seed))))])
  (values X labels))
