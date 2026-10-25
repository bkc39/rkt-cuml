#lang racket/base

(require (only-in cuml/manifold
                  umap-fit
                  umap-params
                  umap-result-a
                  umap-result-b
                  umap-result-embedding
                  umap-result-params
                  umap-transform)
         (only-in racket/list take)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* list->device-vector)
         (only-in "private/python-env.rkt" run-twin test-twin twin-params)
         (only-in "private/umap-data.rkt"
                  bits-equal?
                  far-blobs
                  fixture
                  fixture-X
                  fixture-labels
                  fixture-rows
                  fixture-y
                  held-out-fit
                  held-out-new
                  overlap
                  twin-case))

(define-runtime-path twin "python/umap_twin.py")

(define (ours f options #:y [y #f])
  (define result
    (keyword-apply umap-fit
                   (map car options)
                   (map cdr options)
                   (list (fixture-X f))
                   #:y y))
  (list result
        (device-matrix->list* (umap-result-embedding result))
        (device-matrix->list* (umap-transform result (fixture-X held-out-new)))))

(define (case-of f result #:y [y #f])
  (twin-case f
             (umap-result-params result)
             (umap-result-a result)
             (umap-result-b result)
             #:y y
             #:transform (fixture-rows held-out-new)))

(define configs
  (append (for*/list ([init (in-list '(spectral random))]
                      [metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))])
            (list (cons '#:init init) (cons '#:metric metric) (cons '#:n-neighbors 16)))
          (for*/list ([init (in-list '(spectral random))]
                      [k (in-list '(2 3))])
            (list (cons '#:init init) (cons '#:n-neighbors k)))))

(test-twin "L6: fit and transform equal Python cuML's bitwise on held-out blobs"
  (define runs (for/list ([options (in-list configs)]) (ours held-out-fit options)))
  (define theirs
    (hash-ref (run-twin twin
                        (hasheq 'cases
                                (for/list ([run (in-list runs)])
                                  (case-of held-out-fit (car run)))))
              'results))
  (for ([options (in-list configs)]
        [run (in-list runs)]
        [found (in-list theirs)])
    (match-define (list _ embedding transformed) run)
    (test-case (format "~a" options)
      (check-true (bits-equal? embedding (hash-ref found 'embedding)) "fit")
      (check-true (bits-equal? transformed (hash-ref found 'transform)) "transform"))))

(test-twin "L6: a supervised fit equals Python's given the dense codes"
  (define labels (fixture-labels held-out-fit))
  (define runs
    (for/list ([init (in-list '(spectral random))])
      (ours held-out-fit (list (cons '#:init init)) #:y (fixture-y held-out-fit))))
  (define theirs
    (hash-ref (run-twin twin
                        (hasheq 'cases
                                (for/list ([run (in-list runs)])
                                  (case-of held-out-fit (car run) #:y labels))))
              'results))
  (for ([run (in-list runs)]
        [found (in-list theirs)])
    (check-true (bits-equal? (cadr run) (hash-ref found 'embedding)))))

(define merged-labels '(0 16777216 16777217))

(test-twin "L10: labels past 2^24 stay distinct, where Python merges them"
  (define labels
    (for/list ([i (in-range 600)])
      (list-ref merged-labels (modulo i 3))))
  (define result
    (umap-fit (fixture-X overlap) #:y (list->device-vector labels #:dtype 'int64) #:init 'random))
  (define embedding (device-matrix->list* (umap-result-embedding result)))
  (define (twin-of y)
    (twin-case overlap (umap-result-params result) (umap-result-a result) (umap-result-b result) #:y y))
  (match-define (list dense raw)
    (hash-ref (run-twin twin
                        (hasheq 'cases
                                (list (twin-of (for/list ([i (in-range 600)]) (modulo i 3)))
                                      (twin-of labels))))
              'results))
  (check-true (bits-equal? embedding (hash-ref dense 'embedding)) "dense codes")
  (check-false (bits-equal? embedding (hash-ref raw 'embedding)) "raw labels"))

(define twelve
  (fixture "twelve" (take (fixture-rows far-blobs) 12) (take (fixture-labels far-blobs) 12)))

(test-twin "L10: n-neighbors past the row count is Python's clamp, bitwise"
  (define result (umap-fit (fixture-X twelve) #:n-neighbors 20 #:init 'random))
  (define clamped (umap-fit (fixture-X twelve) #:n-neighbors 12 #:init 'random))
  (match-define (list found)
    (hash-ref (run-twin twin
                        (hasheq 'cases
                                (list (twin-case twelve
                                                 (umap-result-params result)
                                                 (umap-result-a result)
                                                 (umap-result-b result)))))
              'results))
  (define embedding (device-matrix->list* (umap-result-embedding result)))
  (check-true (bits-equal? embedding (device-matrix->list* (umap-result-embedding clamped))))
  (check-true (bits-equal? embedding (hash-ref found 'embedding)))
  (check-equal? (hash-ref found 'n_neighbors) 12))

(define unbound
  '(set-op-mix-ratio local-connectivity
                     repulsion-strength
                     transform-queue-size
                     target-n-neighbors
                     target-weight
                     target-metric
                     hash-input
                     force-serial-epochs
                     callback
                     metric-kwds
                     precomputed-knn
                     build-kwds
                     device-ids))

(test-twin "L15: the parameters are Python's, but for random-state and build-algo"
  (match-define (list theirs)
    (hash-ref (run-twin twin (hasheq 'cases (list (hasheq 'op "params")))) 'results))
  (check-equal? (twin-params theirs unbound)
                (hash-set* (umap-params) 'random-state #f 'build-algo 'auto)))
