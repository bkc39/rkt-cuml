#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base" "cuml-core-lib" "cuml-cluster-lib"))
(define implies '("cuml-core-lib" "cuml-cluster-lib"))
(define pkg-desc "Racket bindings to NVIDIA cuML (no documentation)")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
