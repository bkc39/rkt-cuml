#lang racket/base

(require (only-in cuml/cluster hdbscan-fit hdbscan-result-labels)
         (only-in cuml/manifold umap-fit umap-result-embedding)
         (only-in cuml/metrics adjusted-rand-score)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/umap-data.rkt" far-blobs fixture-X fixture-y))

(test-gpu "L8: HDBSCAN on the far-blobs embedding recovers the groups for seeds 0-9"
  (define X (fixture-X far-blobs))
  (define truth (fixture-y far-blobs))
  (for* ([options (in-list '((spectral brute-force) (random brute-force) (spectral nn-descent)))]
         [seed (in-range 10)])
    (match-define (list init build-algo) options)
    (test-case (format "~a ~a seed ~a" init build-algo seed)
      (check-equal? (~> (umap-fit X #:init init #:build-algo build-algo #:random-state seed)
                        umap-result-embedding
                        (hdbscan-fit #:min-cluster-size 20)
                        hdbscan-result-labels
                        (adjusted-rand-score truth _))
                    1.0))))
