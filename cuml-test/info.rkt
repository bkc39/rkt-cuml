#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps
  '("at-exp-lib" "cuml-cluster-lib"
                 "cuml-core-lib"
                 "cuml-decomposition-lib"
                 "cuml-doc"
                 "cuml-lib"
                 "cuml-manifold-lib"
                 "cuml-metrics-lib"
                 "cuml-preprocessing-lib"
                 "datasets-core"
                 "pict-lib"
                 ("raft" #:version "0.2")
                 "rackunit-lib"
                 "sandbox-lib"
                 "threading-lib"))
(define pkg-desc "Tests for cuml")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
