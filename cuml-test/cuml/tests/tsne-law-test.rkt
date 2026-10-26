#lang racket/base

(require (only-in cuml/cluster hdbscan-fit hdbscan-result-labels)
         (only-in cuml/manifold tsne-fit tsne-result-embedding)
         (only-in cuml/metrics adjusted-rand-score)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/tsne-data.rkt" embedded)
         (only-in "private/umap-data.rkt" far-blobs fixture-X fixture-y))

(test-gpu "L7: HDBSCAN on the far-blobs embedding recovers the groups for seeds 0-9"
  (define X (fixture-X far-blobs))
  (define truth (fixture-y far-blobs))
  (for* ([options (in-list '((fft random) (fft pca) (exact random) (exact pca)))]
         [seed (in-range 10)])
    (match-define (list method init) options)
    (test-case (format "~a ~a seed ~a" method init seed)
      (check-equal? (~> (tsne-fit X #:method method #:init init #:random-state seed)
                        tsne-result-embedding
                        (hdbscan-fit #:min-cluster-size 20)
                        hdbscan-result-labels
                        (adjusted-rand-score truth _))
                    1.0))))

(test-gpu "L9: seeds 0 and 1 differ, and #f draws from the reseeded PRNG"
  (define X (fixture-X far-blobs))
  (check-false (bits-equal? (embedded (tsne-fit X #:random-state 0))
                            (embedded (tsne-fit X #:random-state 1))))
  (define (reseeded seed)
    (random-seed seed)
    (embedded (tsne-fit X #:random-state #f)))
  (check-true (bits-equal? (reseeded 7) (reseeded 7)))
  (check-false (bits-equal? (reseeded 7) (reseeded 8)) "#f draws its seed from the PRNG"))
