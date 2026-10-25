#lang info

#|review: ignore|#

(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps
  '("cuml-lib" "cuml-preprocessing-lib"
               "datasets-core"
               ("raft" #:version "0.2")
               "pict-doc"
               "pict-lib"
               "plot-doc"
               "plot-gui-lib"
               "plot-lib"
               "racket-doc"
               "sandbox-lib"
               "scribble-lib"
               "threading-doc"
               "threading-lib"))
(define update-implies '("cuml-lib"))
(define pkg-desc "Documentation for cuml, the Racket binding to NVIDIA cuML")
(define pkg-authors '(bkc))
(define license 'Apache-2.0)
