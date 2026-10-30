#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-data.rkt" check-quality)
         (only-in "private/umap-data.rkt" digits))

(test-twin "L6: FFT trustworthiness is above its measured floor on digits"
  (check-quality 'fft (list (digits))))
