#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-data.rkt" check-quality)
         (only-in "private/umap-data.rkt" digits))

(test-twin "L6: exact trustworthiness lies within Python cuML's seed spread on digits"
  (check-quality 'exact (list (digits))))
