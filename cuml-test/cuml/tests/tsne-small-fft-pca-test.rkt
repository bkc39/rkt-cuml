#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/tsne-data.rkt" check-small-sweep))

(test-gpu "L3: fft, pca init: n = 2 to 40 give a finite embedding or the divergence refusal"
  (check-small-sweep 'fft 'pca))
