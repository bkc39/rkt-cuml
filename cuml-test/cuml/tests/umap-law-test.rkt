#lang racket/base

(require (only-in cuml/cluster hdbscan-fit hdbscan-result-labels)
         (only-in cuml/manifold umap-fit umap-result-embedding umap-transform)
         (only-in cuml/metrics adjusted-rand-score)
         (only-in rackunit check-false check-true test-case)
         (only-in raft device-matrix->list* list*->device-matrix list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/umap-data.rkt"
                  far-blobs
                  finite-rows?
                  fixture-X
                  fixture-labels
                  fixture-rows
                  fixture-y
                  held-out-fit
                  held-out-new
                  overlap
                  relabel
                  tiny))

(define (embedded result)
  (device-matrix->list* (umap-result-embedding result)))

(define (ari truth E)
  (~> E (hdbscan-fit #:min-cluster-size 20) hdbscan-result-labels (adjusted-rand-score truth _)))

(test-gpu "L3: every embedding and transform is finite"
  (for* ([metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))]
         [init (in-list '(spectral random))])
    (test-case (format "~a ~a" metric init)
      (define result (umap-fit (fixture-X held-out-fit) #:metric metric #:init init))
      (check-true (finite-rows? (embedded result)) "fit")
      (check-true (finite-rows? (device-matrix->list* (umap-transform result (fixture-X held-out-new))))
                  "transform"))))

(test-gpu "L9: supervision does not lower the label agreement on overlap"
  (define X (fixture-X overlap))
  (define truth (fixture-y overlap))
  (define supervised (ari truth (umap-result-embedding (umap-fit X #:y truth))))
  (define unsupervised (ari truth (umap-result-embedding (umap-fit X))))
  (check-true (>= supervised unsupervised) (format "~a < ~a" supervised unsupervised)))

(test-gpu "L9: relabelled y gives the same embedding, and an all -1 y is accepted"
  (define X (fixture-X overlap))
  (define labels (fixture-labels overlap))
  (check-true (bits-equal? (embedded (umap-fit X #:y (fixture-y overlap)))
                           (embedded (umap-fit X
                                               #:y (list->device-vector (relabel labels)
                                                                        #:dtype 'int64)))))
  (define unlabelled (list->device-vector (map (lambda (_) -1) labels) #:dtype 'int32))
  (check-true (finite-rows? (embedded (umap-fit X #:y unlabelled)))))

(test-gpu "L12: transforming the training matrix equals transforming a copy of it"
  (define X (fixture-X held-out-fit))
  (define result (umap-fit X))
  (check-true (bits-equal? (device-matrix->list* (umap-transform result X))
                           (~> (fixture-rows held-out-fit)
                               (list*->device-matrix #:dtype 'float32)
                               (umap-transform result _)
                               device-matrix->list*))))

(test-gpu "L13: seeds 0 and 1 differ, and a reseeded #f random-state repeats below 300 rows"
  (define X (fixture-X far-blobs))
  (for ([init (in-list '(spectral random))])
    (check-false (bits-equal? (embedded (umap-fit X #:init init #:random-state 0))
                              (embedded (umap-fit X #:init init #:random-state 1)))
                 (format "~a" init)))
  (define (reseeded)
    (random-seed 7)
    (embedded (umap-fit (fixture-X tiny) #:random-state #f #:n-neighbors 4)))
  (check-true (bits-equal? (reseeded) (reseeded))))
