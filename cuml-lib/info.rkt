#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps
  '("base" ("raft" #:version "0.2")
           "cuml-core-lib"
           "cuml-cluster-lib"
           "cuml-metrics-lib"
           ("datasets-core" #:version "0.1")))
(define implies '("cuml-core-lib" "cuml-cluster-lib" "cuml-metrics-lib"))
(define pkg-desc "Racket bindings to NVIDIA cuML (no documentation)")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
