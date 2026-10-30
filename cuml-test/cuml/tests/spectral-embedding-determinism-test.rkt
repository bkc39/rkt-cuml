#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-embedding-determinism-data.rkt"
                  check-spectral-embedding-repeatable
                  dense-configs
                  graph-configs)
         (only-in "private/umap-data.rkt" digits fixture-rows))

(test-gpu "L4: dense fits on blobs at 10 and 256 neighbours, and graph fits on sbm and three-blocks, repeat bitwise in and across processes"
  (check-spectral-embedding-repeatable
   (append (dense-configs 'blobs) (graph-configs 'sbm) (graph-configs 'three-blocks))))

(test-twin "L4: dense fits on digits repeat bitwise in and across processes, at 10 and 256 neighbours"
  (check-spectral-embedding-repeatable (dense-configs 'given) #:given (fixture-rows (digits))))
