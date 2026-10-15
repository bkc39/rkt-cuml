#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("cuml-lib" "cuml-doc"))
(define implies '("cuml-lib" "cuml-doc"))
(define pkg-desc "Racket bindings to NVIDIA cuML: GPU machine learning on RAFT device arrays")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
