#lang racket/base

(require (only-in cuml/cluster
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-result-labels
                  spectral-clustering-result?)
         (only-in racket/function curry)
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
                  graph
                  graph->vectors
                  graph-edit
                  graph-labels
                  graph-map-vals
                  graph-n
                  graph-scale-upper
                  graph-upper
                  labels
                  planted?
                  repeat-each
                  set-entry
                  three-blobs-matrix
                  three-blocks
                  unit-path))

(define eigensolver #rx"^spectral-clustering-fit(-graph)?: .*eigensolver couldn't converge")

(test-gpu "SC1: labels are int32, one per sample, each in [0, k), on both paths"
  (define X (three-blobs-matrix))
  (define-values (g _gtruth) (three-blocks))
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
  (define X (three-blobs-matrix))
  (define-values (g _gtruth) (three-blocks))
  (check-equal? (labels X #:n-clusters 1) (make-list 300 0))
  (check-equal? (graph-labels g #:n-clusters 1) (make-list 120 0)))

(test-gpu "SC11: n-neighbors past n-1 counts as n-1, and auto components are min(k, (n-1)/3)"
  (define-values (rows _truth) (blobs 10 2 2 50.0 3))
  (define X (mat rows 'float32))
  (define at-most (labels X #:n-clusters 2 #:n-neighbors 19))
  (for ([nn (in-list '(20 21 1000 2147483647))])
    (check-equal? (labels X #:n-clusters 2 #:n-neighbors nn) at-most (format "~a neighbours" nn)))
  (for ([k (in-list '(2 6 8 20))])
    (check-equal? (labels X #:n-clusters k #:n-init 1)
                  (labels X #:n-clusters k #:n-init 1 #:n-components (min k 6)))
    (check-true
     (planted? (labels X #:n-clusters k #:n-init 1 #:n-neighbors 5)
               (labels X #:n-clusters k #:n-init 1 #:n-neighbors 5 #:n-components (min k 6)))))
  (define-values (g _gtruth) (three-blocks))
  (for ([k (in-list '(3 8))])
    (check-true (planted? (graph-labels g #:n-clusters k #:n-init 1)
                          (graph-labels g #:n-clusters k #:n-init 1 #:n-components k)))))

(define two-triangles
  (graph '(0 1 0 2 1 2 3 4 3 5 4 5 2 3)
         '(1 0 2 0 2 1 4 3 5 3 5 4 3 2)
         '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.01 0.01)
         6))

(define four-path (unit-path 4))

(test-gpu "SC13: below 7 samples the automatic embedding has one component"
  (for* ([g (in-list (list two-triangles four-path))]
         [k (in-list '(2 3))]
         [seed (in-range 3)])
    (test-case (format "~a samples, k ~a, seed ~a" (graph-n g) k seed)
      (check-equal? (graph-labels g #:n-clusters k #:random-state seed)
                    (graph-labels g #:n-clusters k #:random-state seed #:n-components 1))))
  (test-case "a connected graph, split into pieces that are not connected"
    (check-equal? (graph-labels (unit-path 5) #:n-clusters 2 #:random-state 0) '(1 1 1 0 1)))
  (test-case "a connected graph, fewer clusters than asked"
    (check-equal? (graph-labels four-path #:n-clusters 2 #:random-state 0) '(0 0 0 0)))
  (define-values (triangle-and-point truth) (cliques '(3 1)))
  (define-values (separate-triangles _truth) (cliques '(3 3)))
  (for ([seed (in-range 5)])
    (test-case (format "disconnected groups, seed ~a" seed)
      (check-true (planted? (graph-labels triangle-and-point #:n-clusters 2 #:random-state seed)
                            truth))
      (check-equal? (graph-labels separate-triangles #:n-clusters 2 #:random-state seed)
                    (make-list 6 0)))))

(test-gpu "SC14: weights whose sum passes the largest float32, or subnormal weights, are refused"
  (define-values (g truth) (three-blocks))
  (define (scaled factor)
    (graph-map-vals g (curry * factor)))
  (check-cuml-error
   'logic
   #rx"^spectral-clustering-fit-graph: vals: the weights sum to .*, more than 3.4028234663852886e\\+38$"
   (lambda () (graph-labels (scaled 3e37) #:n-clusters 3)))
  (check-true (planted? (graph-labels (scaled 1e35) #:n-clusters 3) truth))
  (check-cuml-error
   'logic
   #rx"^spectral-clustering-fit-graph: vals: entry 0: expected 0 or a weight of at least"
   (lambda () (graph-labels (graph-edit g #:vals (set-entry 0 1e-44)) #:n-clusters 3)))
  (check-true (planted? (graph-labels (graph-edit g #:vals (set-entry 0 1e-37)) #:n-clusters 3)
                        truth)))

(test-gpu "SC12: a fit leaves X, rows, cols and vals as they were"
  (define X (three-blobs-matrix))
  (define before (device-matrix->list* X))
  (spectral-clustering-fit X #:n-clusters 3)
  (check-equal? (device-matrix->list* X) before)
  (define-values (g _gtruth) (three-blocks))
  (define vectors (graph->vectors g))
  (define graph-before (map device-vector->list vectors))
  (match-define (list rows cols vals) vectors)
  (spectral-clustering-fit-graph rows cols vals (graph-n g) #:n-clusters 3)
  (check-equal? (map device-vector->list vectors) graph-before))

(test-gpu "P1: an upper-triangle graph, or one with its upper triangle down-scaled, does not converge"
  (define-values (g _truth) (three-blocks))
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

(test-gpu "P4: one edge among 1000 samples, the rest isolated, does not converge"
  (define one-edge (graph '(0 1) '(1 0) '(1.0 1.0) 1000))
  (for ([k (in-list '(1 2 3))])
    (check-cuml-error 'generic eigensolver (lambda () (graph-labels one-edge #:n-clusters k)))))
