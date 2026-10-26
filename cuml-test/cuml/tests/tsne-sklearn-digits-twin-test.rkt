#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-data.rkt" check-sklearn)
         (only-in "private/umap-data.rkt" digits))

(test-twin "L6: exact trustworthiness lies within 0.05 of sklearn's exact t-SNE on digits"
  (check-sklearn (list (digits))))
