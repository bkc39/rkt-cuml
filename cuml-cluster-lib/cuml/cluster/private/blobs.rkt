#lang racket/base

(require (only-in raft current-device-resources device-matrix device-vector)
         (only-in raft/unsafe with-array-views)
         (only-in "native.rkt" call/cluster cuml-cluster-make-blobs cuml-cluster-make-blobs-job))

(provide blobs
         blobs/short-call)

(define shuffle 1)
(define center-box-min -10.0)
(define center-box-max 10.0)

(define (blobs-outputs n-samples n-features element-type resources)
  (values (device-matrix n-samples n-features #:dtype element-type #:resources resources)
          (device-vector n-samples #:dtype 'int32 #:resources resources)))

(define (blobs n-samples
               n-features
               #:centers [centers 3]
               #:cluster-std [cluster-std 1.0]
               #:seed [seed 0]
               #:dtype [element-type 'float32]
               #:resources [resources (current-device-resources)])
  (define-values (X labels) (blobs-outputs n-samples n-features element-type resources))
  (cuml-cluster-make-blobs-job 'blobs
                               resources
                               X
                               labels
                               centers
                               (exact->inexact cluster-std)
                               shuffle
                               center-box-min
                               center-box-max
                               seed)
  (values X labels))

(define (blobs/short-call n-samples
                          n-features
                          #:centers [centers 3]
                          #:cluster-std [cluster-std 1.0]
                          #:seed [seed 0]
                          #:dtype [element-type 'float32]
                          #:resources [resources (current-device-resources)])
  (define-values (X labels) (blobs-outputs n-samples n-features element-type resources))
  (with-array-views #:resources [handle resources] ([x X] [y labels])
    (call/cluster 'blobs
                  (lambda ()
                    (cuml-cluster-make-blobs handle
                                             x
                                             y
                                             centers
                                             (exact->inexact cluster-std)
                                             shuffle
                                             center-box-min
                                             center-box-max
                                             seed))))
  (values X labels))
