#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/tsne-data.rkt" check-quality)
         (only-in "private/umap-data.rkt" blobs overlap))

(test-gpu "L6: FFT trustworthiness is above its measured floor on blobs and overlap"
  (check-quality 'fft (list blobs overlap)))
