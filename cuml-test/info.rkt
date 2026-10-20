#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps
  '("cuml-cluster-lib" "cuml-core-lib"
                       "cuml-lib"
                       "datasets"
                       "datasets-core"
                       ("raft" #:version "0.2")
                       "rackunit-lib"
                       "threading-lib"))
(define pkg-desc "Tests for cuml")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
