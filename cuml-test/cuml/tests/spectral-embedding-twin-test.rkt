#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-data.rkt" three-blocks)
         (only-in "private/spectral-embedding-data.rkt" check-graph-parity check-se-parity sbm)
         (only-in "private/spectral-embedding-twin-data.rkt" parity-configs)
         (only-in "private/umap-data.rkt" blobs fixture-rows))

(test-twin "L5: dense fits on blobs are Python cuML's, bitwise, at 10 and 256 neighbours"
  (for ([n-neighbors (in-list '(10 256))])
    (check-se-parity "blobs" (fixture-rows blobs) (parity-configs n-neighbors))))

(test-twin "L5: graph fits on sbm, weighted sbm and three-blocks are Python cuML's, bitwise"
  (define-values (planted _p) (sbm '(40 60 80) 0.3 0.01 0))
  (define-values (weighted _w) (sbm '(40 60 80) 0.3 0.01 0 #:weighted? #t))
  (define-values (blocks _b) (three-blocks))
  (check-graph-parity "sbm" planted (parity-configs))
  (check-graph-parity "sbm weighted" weighted (parity-configs))
  (check-graph-parity "three-blocks" blocks (parity-configs)))
