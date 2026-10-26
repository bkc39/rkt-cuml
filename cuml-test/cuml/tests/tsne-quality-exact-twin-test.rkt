#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-data.rkt" check-quality)
         (only-in "private/umap-data.rkt" blobs digits overlap))

(test-twin "L6: exact trustworthiness lies within Python cuML's seed spread and 0.05 of sklearn's"
  (check-quality 'exact (list blobs overlap (digits))))
