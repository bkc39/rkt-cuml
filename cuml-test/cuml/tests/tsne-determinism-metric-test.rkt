#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/tsne-determinism-data.rkt" check-tsne-repeatable))

(test-gpu "L4: FFT fits on far-blobs repeat bitwise in and across processes, the other metrics"
  (check-tsne-repeatable 'far-blobs '(manhattan chebyshev cosine)))
