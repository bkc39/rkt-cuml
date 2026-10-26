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
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/umap-data.rkt"
                  digits
                  far-blobs
                  fixture
                  fixture-X
                  fixture-labels
                  fixture-name
                  fixture-rows
                  held-out-fit
                  held-out-new
                  overlap
                  tail-split
                  twin-case))

(define-runtime-path twin "python/umap_twin.py")

(define (ours f new options #:y [y #f])
  (define result
    (keyword-apply umap-fit (map car options) (map cdr options) (list (fixture-X f)) #:y y))
  (list result
        (device-matrix->list* (umap-result-embedding result))
        (device-matrix->list* (umap-transform result (fixture-X new)))))

(define (case-of f new result #:y [y #f])
  (twin-case f
             (umap-result-params result)
             (umap-result-a result)
             (umap-result-b result)
             #:y y
             #:transform (fixture-rows new)))

(define (check-parity f new configs)
  (define runs
    (for/list ([options (in-list configs)])
      (ours f new options)))
  (define theirs
    (hash-ref (run-twin twin
                        (hasheq 'cases (for/list ([run (in-list runs)])
                                         (case-of f new (car run)))))
              'results))
  (for ([options (in-list configs)]
        [run (in-list runs)]
        [found (in-list theirs)])
    (match-define (list _ embedding transformed) run)
    (test-case (format "~a ~a" (fixture-name f) options)
      (check-true (bits-equal? embedding (hash-ref found 'embedding)) "fit")
      (check-true (bits-equal? transformed (hash-ref found 'transform)) "transform"))))

(define configs
  (append (for*/list ([init (in-list '(spectral random))]
                      [metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))])
            (list (cons '#:init init) (cons '#:metric metric) (cons '#:n-neighbors 16)))
          (for*/list ([init (in-list '(spectral random))]
                      [k (in-list '(3 16))])
            (list (cons '#:init init) (cons '#:n-components k)))
          (list (list (cons '#:init 'random)
                      (cons '#:learning-rate 0.5)
                      (cons '#:min-dist 0.3)
                      (cons '#:n-epochs 60)
                      (cons '#:negative-sample-rate 2)))))

(test-twin "L6: fit and transform equal Python cuML's bitwise on held-out blobs"
  (check-parity held-out-fit held-out-new configs))

(test-twin "L6: fit and transform equal Python cuML's bitwise on digits"
  (define-values (head tail) (tail-split (digits)))
  (check-parity head
                tail
                (for*/list ([init (in-list '(spectral random))]
                            [k (in-list '(2 3 16))])
                  (list (cons '#:init init) (cons '#:n-components k)))))

(define (partly-unlabelled labels)
  (for/list ([label (in-list labels)]
             [i (in-naturals)])
    (if (zero? (modulo i 3)) -1 label)))

(test-twin "L6: a supervised fit equals Python's given the dense codes, -1 unlabelled"
  (define labels (fixture-labels held-out-fit))
  (define partial (partly-unlabelled labels))
  (define cases
    (list (cons 'spectral labels) (cons 'random labels) (cons 'random partial)))
  (define runs
    (for/list ([c (in-list cases)])
      (match-define (cons init y) c)
      (ours held-out-fit
            held-out-new
            (list (cons '#:init init))
            #:y (list->device-vector y #:dtype 'int32))))
  (define theirs
    (hash-ref (run-twin twin
                        (hasheq 'cases (for/list ([run (in-list runs)]
                                                  [c (in-list cases)])
                                         (case-of held-out-fit held-out-new (car run) #:y (cdr c)))))
              'results))
  (for ([run (in-list runs)]
        [c (in-list cases)]
        [found (in-list theirs)])
    (match-define (list _ embedding _) run)
    (check-true (bits-equal? embedding (hash-ref found 'embedding)) (format "~a" (car c)))))

(define merged-labels '(0 16777216 16777217))

(test-twin "L10: labels past 2^24 stay distinct, where Python merges them"
  (define labels
    (for/list ([_ (in-range 600)]
               [label (in-cycle merged-labels)])
      label))
  (define result
    (umap-fit (fixture-X overlap) #:y (list->device-vector labels #:dtype 'int64) #:init 'random))
  (define embedding (device-matrix->list* (umap-result-embedding result)))
  (define (twin-of y)
    (twin-case overlap
               (umap-result-params result)
               (umap-result-a result)
               (umap-result-b result)
               #:y y))
  (match-define (list dense raw)
    (hash-ref (run-twin twin
                        (hasheq 'cases (list (twin-of (for/list ([i (in-range 600)])
                                                        (modulo i 3)))
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
                        (hasheq 'cases (list (twin-case twelve
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
