#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-data.rkt" check-quality)
         (only-in "private/umap-data.rkt" blobs digits overlap))

(test-twin "L6: FFT trustworthiness lies within Python cuML's seed spread on blobs, overlap and digits"
  (check-quality 'fft (list blobs overlap (digits))))
