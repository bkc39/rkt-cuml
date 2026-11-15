#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps
  '("cuml-linear-model-lib" "cuml-test"
                            "glmnet"
                            ("raft" #:version "0.2")
                            "rackunit-lib"
                            "threading-lib"))
(define pkg-desc "cuml's parity tests against glmnet, apart from cuml-test for glmnet's licence")
(define pkg-authors '(bkc))
(define license 'GPL-2.0-or-later)
