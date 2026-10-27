#lang racket/base

(require (only-in cuml/manifold spectral-embedding-fit-graph spectral-embedding-result?)
         (only-in racket/function const)
         (only-in racket/list make-list)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-pred test-case)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/spectral-data.rkt"
                  graph
                  graph->vectors
                  graph-edit
                  graph-map-vals
                  set-entry)
         (only-in "private/spectral-embedding-data.rkt" embedded-graph))

(define ring-n 10)

(define ring
  (let ([pairs (for*/list ([i (in-range ring-n)]
                           [e (in-list (list (cons i (modulo (add1 i) ring-n))
                                             (cons (modulo (add1 i) ring-n) i)))])
                 e)])
    (graph (map car pairs) (map cdr pairs) (make-list (length pairs) 1.0) ring-n)))

(define pool-device 0)

(define (fit-vectors rows cols vals [n ring-n])
  (spectral-embedding-fit-graph rows cols vals n))

(define (fault g [options (hasheq)])
  (lambda () (embedded-graph g options)))

(define (with-vectors f)
  (lambda () (match (graph->vectors ring) [(list rows cols vals) (f rows cols vals)])))

(define (index-message name entry got)
  (format "~a: entry ~a: expected an index in [0, 10), got ~a" name entry got))

(define past-sum (const (exact->inexact (expt 2 125))))

(define (graph-rows-of)
  (list
   (row "s1" "n-components: expected at least 1, got 0" (fault ring (hasheq 'n-components 0)))
   (row "g1 rows float32"
        "rows: expected int32, got float32"
        (with-vectors (lambda (_ cols vals) (fit-vectors (blank-vector 20 'float32) cols vals))))
   (row "g1 vals float64"
        "vals: expected float32, got float64"
        (with-vectors (lambda (rows cols _) (fit-vectors rows cols (blank-vector 20 'float64)))))
   (row "g1 rank"
        "rows: expected rank 1, got rank 2"
        (with-vectors (lambda (_ cols vals) (fit-vectors (blank-matrix 20 1 'int32) cols vals))))
   (row "g1 cols"
        "cols: expected 20 entries, as rows has, got 19"
        (with-vectors (lambda (rows _ vals) (fit-vectors rows (blank-vector 19 'int32) vals))))
   (row "g2"
        "n-components: 2 components with drop-first need at least 8 samples, got 7"
        (with-vectors (lambda (rows cols vals) (fit-vectors rows cols vals 7))))
   (row "g2 most"
        "n-samples: expected at most 2147483647, got 2147483648"
        (with-vectors (lambda (rows cols vals) (fit-vectors rows cols vals (expt 2 31)))))
   (row "g5 index" (index-message "rows" 7 10) (fault (graph-edit ring #:rows (set-entry 7 10))))
   (row "g5 index -1" (index-message "cols" 7 -1) (fault (graph-edit ring #:cols (set-entry 7 -1))))
   (row "g5 self-loop"
        "rows: entry 12: self-loop at sample 6; leave the diagonal out"
        (fault (graph-edit ring #:cols (set-entry 12 6))))
   (row "g5 negative"
        "vals: entry 1: expected a finite weight of at least 0, got -1"
        (fault (graph-edit ring #:vals (set-entry 1 -1.0))))
   (row "g5 nan"
        "vals: entry 3: expected a finite weight of at least 0, got nan"
        (fault (graph-edit ring #:vals (set-entry 3 +nan.0))))
   (row "g5 subnormal"
        (string-append "vals: entry 3: expected 0 or a weight of at least 1.1754943508222875e-38,"
                       " got 9.80908925027372e-45")
        (fault (graph-edit ring #:vals (set-entry 3 1e-44))))
   (row "g6"
        "vals: the weights sum to 8.507059173023462e+38, more than 3.4028234663852886e+38"
        (fault (graph-map-vals ring past-sum)))
   (row "g7" "vals: expected at least one positive weight, got none" (fault (graph-map-vals ring (const 0.0))))
   (row "g1 before g2"
        "cols: expected 20 entries, as rows has, got 19"
        (with-vectors (lambda (rows _ vals) (fit-vectors rows (blank-vector 19 'int32) vals 7))))
   (row "g2 before g5"
        "n-components: 2 components with drop-first need at least 8 samples, got 7"
        (lambda ()
          (match (graph->vectors (graph-edit ring #:cols (set-entry 12 6)))
            [(list rows cols vals) (fit-vectors rows cols vals 7)])))
   (row "g5 before g6"
        "vals: entry 1: expected a finite weight of at least 0, got -1"
        (fault (graph-edit (graph-map-vals ring past-sum) #:vals (set-entry 1 -1.0))))))

(test-gpu "L13: fit-graph refuses each bad graph with its message, the first refusal winning"
  (for ([r (in-list (graph-rows-of))])
    (test-case (row-id r)
      (check-row 'spectral-embedding-fit-graph r))))

(test-gpu "L13: g2: the least samples fit, and a valid call succeeds after a refusal"
  (for ([r (in-list (graph-rows-of))])
    (check-row 'spectral-embedding-fit-graph r)
    (check-pred spectral-embedding-result? ((fault ring)) (row-id r)))
  (check-pred spectral-embedding-result? ((fault ring (hasheq 'n-components 3 'drop-first #f)))))

(test-pools "L13: a graph refusal leaves the RMM pool where it was"
  (for ([r (in-list (graph-rows-of))])
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (check-row 'spectral-embedding-fit-graph r)
    (check-equal? (probe-pool-used pool-device) before (row-id r))))
