#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/spectral-embedding-determinism-data.rkt"
                  check-se-repeatable
                  dense-configs
                  graph-configs))

(test-gpu "L4: dense fits on blobs repeat bitwise in and across processes, at 10 and 256 neighbours"
  (check-se-repeatable (dense-configs 'blobs)))

(test-gpu "L4: graph fits on sbm and three-blocks repeat bitwise in and across processes"
  (check-se-repeatable (append (graph-configs 'sbm) (graph-configs 'three-blocks))))
