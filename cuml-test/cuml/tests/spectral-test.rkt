#lang racket/base

(require (only-in cuml/cluster
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-result-labels
                  spectral-clustering-result?)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/spectral-data.rkt"
                  blobs
                  cliques
                  graph->vectors
                  graph-labels
                  graph-n
                  graph-scale-upper
                  graph-upper
                  labels
                  planted?
                  repeat-each
                  sparse-blocks))

(define eigensolver #rx"^spectral-clustering-fit(-graph)?: .*eigensolver couldn't converge")

(define (three-blobs)
  (define-values (rows truth) (blobs 100 3 2 50.0 0))
  (values (mat rows 'float32) truth))

(define (blocks)
  (sparse-blocks '(30 40 50) 0.3 0))

(test-gpu "SC1: labels are int32, one per sample, each in [0, k), on both paths"
  (define-values (X _truth) (three-blobs))
  (define-values (g _gtruth) (blocks))
  (match-define (list rows cols vals) (graph->vectors g))
  (for ([k (in-list '(2 3 5 8))])
    (test-case (format "k ~a" k)
      (define dense (spectral-clustering-fit X #:n-clusters k))
      (define sparse (spectral-clustering-fit-graph rows cols vals (graph-n g) #:n-clusters k))
      (for ([r (in-list (list dense sparse))]
            [n (in-list '(300 120))])
        (check-true (spectral-clustering-result? r))
        (define found (spectral-clustering-result-labels r))
        (check-equal? (list (dtype found) (shape found)) (list 'int32 (list n)))
        (check-true (for/and ([l (in-list (device-vector->list found))])
                      (< -1 l k)))))))

(test-gpu "SC2: one cluster labels every sample 0"
  (define-values (X _truth) (three-blobs))
  (define-values (g _gtruth) (blocks))
  (check-equal? (labels X #:n-clusters 1) (make-list 300 0))
  (check-equal? (graph-labels g #:n-clusters 1) (make-list 120 0)))

(test-gpu "SC11: n-neighbors past n-1 counts as n-1, and auto components are min(k, (n-1)/3)"
  (define-values (rows _truth) (blobs 10 2 2 50.0 3))
  (define X (mat rows 'float32))
  (define at-most (labels X #:n-clusters 2 #:n-neighbors 19))
  (for ([nn (in-list '(20 21 1000 2147483647))])
    (check-equal? (labels X #:n-clusters 2 #:n-neighbors nn) at-most (format "~a neighbours" nn)))
  (for ([k (in-list '(2 6 8 20))])
    (check-equal? (labels X #:n-clusters k) (labels X #:n-clusters k #:n-components (min k 6)))
    (check-true (planted? (labels X #:n-clusters k #:n-neighbors 5)
                          (labels X #:n-clusters k #:n-neighbors 5 #:n-components (min k 6)))))
  (define-values (g _gtruth) (blocks))
  (for ([k (in-list '(3 8))])
    (check-true (planted? (graph-labels g #:n-clusters k)
                          (graph-labels g #:n-clusters k #:n-components k)))))

(test-gpu "SC12: a fit leaves X, rows, cols and vals as they were"
  (define-values (X _truth) (three-blobs))
  (define before (device-matrix->list* X))
  (spectral-clustering-fit X #:n-clusters 3)
  (check-equal? (device-matrix->list* X) before)
  (define-values (g _gtruth) (blocks))
  (define vectors (graph->vectors g))
  (define graph-before (map device-vector->list vectors))
  (match-define (list rows cols vals) vectors)
  (spectral-clustering-fit-graph rows cols vals (graph-n g) #:n-clusters 3)
  (check-equal? (map device-vector->list vectors) graph-before))

(test-gpu "P1: an upper-triangle graph, or one with its upper triangle down-scaled, does not converge"
  (define-values (g _truth) (blocks))
  (check-cuml-error 'generic eigensolver (lambda () (graph-labels (graph-upper g) #:n-clusters 3)))
  (check-cuml-error 'generic
                    eigensolver
                    (lambda () (graph-labels (graph-scale-upper g 0.2) #:n-clusters 3))))

(test-gpu "P2: cliques of 5, 17 and 30 do not converge at seed 0; three of 10 are recovered"
  (define-values (uneven _uneven-truth) (cliques '(5 17 30)))
  (check-cuml-error 'generic
                    eigensolver
                    (lambda () (graph-labels uneven #:n-clusters 3 #:random-state 0)))
  (define-values (three three-truth) (cliques '(10 10 10)))
  (check-true (planted? (graph-labels three #:n-clusters 3) three-truth)))

(test-gpu "P3: rows repeated 30 times do not converge at seed 0; repeated twice they are recovered"
  (define-values (rows truth) (blobs 40 3 2 50.0 0))
  (check-cuml-error 'generic
                    eigensolver
                    (lambda ()
                      (labels (mat (repeat-each rows 30) 'float32) #:n-clusters 3 #:random-state 0)))
  (check-true (planted? (labels (mat (repeat-each rows 2) 'float32) #:n-clusters 3)
                        (repeat-each truth 2))))
