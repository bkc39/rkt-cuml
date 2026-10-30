#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/umap-data.rkt" blobs check-quality held-out-fit held-out-new overlap))

(test-twin "L7: trustworthiness of fits and transforms is within Python cuML's seed spread"
  (for ([init (in-list '(spectral random))])
    (check-quality init (list blobs overlap) held-out-fit held-out-new)))
