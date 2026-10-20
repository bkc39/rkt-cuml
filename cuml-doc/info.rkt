#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps
  '("cuml-lib" "datasets"
               ("raft" #:version "0.2")
               "racket-doc"
               "sandbox-lib"
               "scribble-lib"
               "threading-lib"))
(define update-implies '("cuml-lib"))
(define pkg-desc "Documentation for cuml, the Racket binding to NVIDIA cuML")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
