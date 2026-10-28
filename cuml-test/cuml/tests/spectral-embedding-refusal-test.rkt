#lang racket/base

(require (only-in cuml fit fit-transform spectral-embedding transform)
         (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-fit-transform
                  spectral-embedding-params
                  spectral-embedding-result?)
         (only-in racket/flonum flsingle)
         (only-in racket/function const)
         (only-in racket/list make-list take)
         (only-in racket/match match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check-equal? check-exn check-pred check-true fail test-case)
         (only-in raft exn:fail:raft? list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/manifold-refusal-data.rkt"
                  large-message
                  large-row
                  listed-X
                  nan-message
                  nan-row
                  tiny-with)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id)
         (only-in "private/scaler-data.rkt" rows->matrix)
         (only-in "private/spectral-data.rkt"
                  graph
                  graph->vectors
                  graph-edit
                  graph-map-vals
                  set-entry)
         (only-in "private/spectral-embedding-data.rkt"
                  embedded
                  embedded-graph
                  heavy-pair
                  sign-ruled?)
         (only-in "private/umap-data.rkt" far-blobs finite-rows? fixture-X fixture-rows tiny))

(define small (hasheq 'n-neighbors 4))

(define pool-device 0)

(define (tiny-X)
  (fixture-X tiny))

(define (valid-X)
  (fixture-X far-blobs))

(define (line count [offset 0.0])
  (for/list ([i (in-range count)])
    (list (+ offset (exact->inexact i)) (exact->inexact (modulo i 3)))))

(define (line-X count)
  (rows->matrix (line count) 'float32))

(define two-message "n-components: 2 components with drop-first need at least 8 rows, got 7")

(define zero-message
  (string-append "embedding: column 0 is zero: the eigensolver found no eigenvector of this graph;"
                 " deduplicate or centre X, or change n-neighbors"))

(define (halves)
  (rows->matrix (append (make-list 100 '(1.0 2.0)) (make-list 100 '(3.0 5.0))) 'float32))

(define (offset)
  (rows->matrix (line 64 1e9) 'float32))

(define (X-row id message X [options small])
  (row id message (cons X options)))

(define (X-rows)
  (define one-column (rows->matrix (map (lambda (r) (take r 1)) (fixture-rows tiny)) 'float32))
  (list
   (X-row "s1" "n-components: expected at least 1, got 0" (tiny-X) (hasheq 'n-components 0))
   (X-row "s2" "n-neighbors: expected at least 2, got 1" (tiny-X) (hasheq 'n-neighbors 1))
   (X-row "s3" "X: expected float32, got float64" (tiny-with '() 'float64))
   (X-row "s4 rank" "X: expected rank 2, got rank 1" (blank-vector 8 'float32))
   (X-row "s4 layout"
          "X: expected row-major, got col-major 8x3"
          (blank-matrix 8 3 'float32 #:layout 'col-major))
   (X-row "s5" "X: expected at least 2 columns, got 1" one-column)
   (X-row "s7" two-message (line-X 7) (hasheq 'n-neighbors 2))
   (X-row "s7 drop-first #f"
          "n-components: 2 components need at least 6 rows, got 5"
          (line-X 5)
          (hasheq 'n-neighbors 2 'drop-first #f))
   (X-row "s7 one component"
          "n-components: 1 component with drop-first needs at least 6 rows, got 5"
          (line-X 5)
          (hasheq 'n-components 1))
   (X-row "s9" "n-neighbors: expected at most the 8 rows, got 9" (tiny-X) (hasheq 'n-neighbors 9))
   (X-row "s10" nan-message (tiny-with nan-row))
   (X-row "s10 inf"
          "X: row 0, column 2: expected a finite value, got -inf"
          (tiny-with '([(0 2) -inf.0])))
   (X-row "s11" large-message (tiny-with large-row))
   (X-row "s12"
          "X: every row equals row 0; a spectral embedding needs two distinct rows"
          (rows->matrix (make-list 8 '(1.5 1.5 1.5)) 'float32))
   (X-row "s14 halves" zero-message (halves) (hasheq 'n-neighbors 10))
   (X-row "s14 offset" zero-message (offset) (hasheq 'n-neighbors 10))
   (X-row "s14 every row a neighbour" zero-message (line-X 50) (hasheq 'n-neighbors 50))
   (X-row "s1 before s3"
          "n-components: expected at least 1, got 0"
          (tiny-with '() 'float64)
          (hasheq 'n-components 0))
   (X-row "s2 before s3"
          "n-neighbors: expected at least 2, got 1"
          (tiny-with '() 'float64)
          (hasheq 'n-neighbors 1))
   (X-row "s5 before s7" "X: expected at least 2 columns, got 1" (blank-matrix 3 1 'float32))
   (X-row "s7 before s9" two-message (line-X 7) (hasheq 'n-neighbors 9))
   (X-row "s10 before s11" nan-message (tiny-with (append nan-row large-row)))
   (X-row "s11 before s12"
          (format "X: row 0: expected a squared norm of at most 4.253529586511731e+37, got ~a"
                  (* 3 (sqr (flsingle 1e20))))
          (rows->matrix (make-list 8 '(1e20 1e20 1e20)) 'float32))))

(define (fit-calls)
  (list (cons 'spectral-embedding-fit
              (lambda (X options) (apply-options spectral-embedding-fit options X)))
        (cons 'spectral-embedding-fit-transform
              (lambda (X options) (apply-options spectral-embedding-fit-transform options X)))
        (cons 'fit (lambda (X options) (fit (apply-options spectral-embedding options) X)))
        (cons 'fit-transform
              (lambda (X options) (fit-transform (apply-options spectral-embedding options) X)))))

(test-gpu "L13: the fits refuse each bad X or parameter with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (X-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (match-define (cons X options) (row-call r))
      (check-row who r (lambda () (call X options)))))
  (for ([bad (in-list (list (lambda () (spectral-embedding-fit (tiny-X) #:n-neighbors 2.5))
                            (lambda () (spectral-embedding-fit (tiny-X) #:n-components 'two))
                            (lambda () (spectral-embedding-fit (tiny-X) #:random-state -1))
                            (lambda ()
                              (spectral-embedding-fit (tiny-X) #:random-state (expt 2 64)))))])
    (check-exn exn:fail:contract? bad)))

(test-case "L13: s0: an affinity other than nearest-neighbors in params is refused before X is read"
  (for ([affinity (in-list '(precomputed rbf))])
    (check-cuml-error
     'logic
     (format "spectral-embedding-fit: affinity: expected nearest-neighbors, got ~a" affinity)
     (lambda ()
       (spectral-embedding-fit 'X
                               #:params (spectral-embedding-params #:affinity affinity)
                               #:n-components 0
                               #:resources 'unused))))
  (check-cuml-error 'logic
                    "fit: affinity: expected nearest-neighbors, got precomputed"
                    (lambda ()
                      (fit (spectral-embedding #:affinity 'precomputed #:resources 'unused) 'X))))

(define (fit-row r)
  (match-define (cons X options) (row-call r))
  (check-row 'spectral-embedding-fit r (lambda () (apply-options spectral-embedding-fit options X))))

(test-gpu "L13: s7: the least rows fit for every probed (n-components, drop-first)"
  (for ([entry (in-list '((1 #f 4 "1 component needs at least 4 rows, got 3")
                          (1 #t 6 "1 component with drop-first needs at least 6 rows, got 5")
                          (2 #t 8 "2 components with drop-first need at least 8 rows, got 7")
                          (2 #f 6 "2 components need at least 6 rows, got 5")
                          (8 #t 20 "8 components with drop-first need at least 20 rows, got 19")
                          (19 #t 42 "19 components with drop-first need at least 42 rows, got 41")))])
    (match-define (list k drop least message) entry)
    (test-case (format "k ~a drop ~a" k drop)
      (define options
        (hasheq 'n-components k 'drop-first drop 'n-neighbors (add1 (quotient least 2))))
      (check-cuml-error 'logic
                        (string-append "spectral-embedding-fit: n-components: " message)
                        (lambda ()
                          (apply-options spectral-embedding-fit options (line-X (sub1 least)))))
      (define rows (embedded (apply-options spectral-embedding-fit options (line-X least))))
      (check-true (and (finite-rows? rows) (sign-ruled? rows))))))

(test-gpu "L13: s14's remedies fit: centred offset rows and fewer neighbours"
  (for ([entry (in-list (list (list (line-X 64) small) (list (line-X 50) (hasheq 'n-neighbors 10))))])
    (match-define (list X options) entry)
    (check-pred spectral-embedding-result? (apply-options spectral-embedding-fit options X))))

(test-pools "L13: s14: a zero column is refused and freed, under every fit"
  (for ([entry (in-list (fit-calls))])
    (match-define (cons who call) entry)
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (check-cuml-error 'logic
                      (format "~a: ~a" who zero-message)
                      (lambda () (call (halves) (hasheq 'n-neighbors 10))))
    (drain-finalizers!)
    (check-equal? (probe-pool-used pool-device) before (symbol->string who))))

(test-gpu "L13: a non-array X is refused by the view check, under any seed"
  (for ([seed (in-list '(0 #f))])
    (match (raised (lambda () (spectral-embedding-fit listed-X #:random-state seed)))
      [(? exn:fail:raft? e)
       (check-equal?
        (exn-message e)
        (format "spectral-embedding-fit: expected a device array or #f for a view, given ~e"
                listed-X))]
      [other (fail (format "expected an exn:fail:raft, got ~e" other))])))

(test-gpu "L13: the generics refuse with their own names"
  (define w (list->device-vector '(1 1 1 1 1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic
                    "transform: spectral-embedding has no transform; use fit-transform"
                    (lambda ()
                      (transform (fit (spectral-embedding #:n-neighbors 10) (valid-X)) (valid-X))))
  (check-cuml-error 'logic
                    "fit: spectral-embedding takes no sample-weight"
                    (lambda () (fit (spectral-embedding) (tiny-X) #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: spectral-embedding takes no sample-weight"
                    (lambda () (fit-transform (spectral-embedding) (tiny-X) #:sample-weight w))))

(test-gpu "L13: after a refused s7, s10, s11, s12 or s14, a valid call succeeds"
  (for ([r (in-list (X-rows))]
        #:when (member (row-id r) '("s7" "s10" "s11" "s12" "s14 halves" "s14 offset")))
    (fit-row r)
    (check-pred spectral-embedding-result?
                (spectral-embedding-fit (valid-X) #:n-neighbors 10)
                (row-id r))))

(test-pools "L13: a refusal leaves the RMM pool where it was"
  (for ([r (in-list (X-rows))])
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (fit-row r)
    (check-equal? (probe-pool-used pool-device) before (row-id r))))

(define ring-n 10)

(define (ring-of n)
  (define pairs
    (for*/list ([i (in-range n)]
                [e (in-list (list (cons i (modulo (add1 i) n)) (cons (modulo (add1 i) n) i)))])
      e))
  (graph (map car pairs) (map cdr pairs) (make-list (length pairs) 1.0) n))

(define ring (ring-of ring-n))

(define (fit-vectors rows cols vals [n ring-n])
  (spectral-embedding-fit-graph rows cols vals n))

(define (fault g [options (hasheq)])
  (lambda () (embedded-graph g options)))

(define (with-vectors f)
  (lambda ()
    (match-define (list rows cols vals) (graph->vectors ring))
    (f rows cols vals)))

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
   (row "g7"
        "vals: expected at least one positive weight, got none"
        (fault (graph-map-vals ring (const 0.0))))
   (row
    "zero column"
    (string-append "embedding: column 0 is zero: the eigensolver found no eigenvector of this graph;"
                   " weights spanning many orders of magnitude need norm-laplacian")
    (fault (heavy-pair 24) (hasheq 'norm-laplacian #f)))
   (row "g1 before g2"
        "cols: expected 20 entries, as rows has, got 19"
        (with-vectors (lambda (rows _ vals) (fit-vectors rows (blank-vector 19 'int32) vals 7))))
   (row "g2 before g5"
        "n-components: 2 components with drop-first need at least 8 samples, got 7"
        (lambda ()
          (match-define (list rows cols vals)
            (graph->vectors (graph-edit ring #:cols (set-entry 12 6))))
          (fit-vectors rows cols vals 7)))
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
  (for ([entry (in-list (list (list (ring-of 8) (hasheq))
                              (list (ring-of 4) (hasheq 'n-components 1 'drop-first #f))
                              (list (ring-of 6) (hasheq 'n-components 1))
                              (list ring (hasheq 'n-components 3 'drop-first #f))))])
    (match-define (list g options) entry)
    (define rows (embedded (embedded-graph g options)))
    (check-true (and (finite-rows? rows) (sign-ruled? rows)) (format "~a samples" (length rows)))))

(test-pools "L13: a graph refusal leaves the RMM pool where it was"
  (for ([r (in-list (graph-rows-of))])
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (check-row 'spectral-embedding-fit-graph r)
    (drain-finalizers!)
    (check-equal? (probe-pool-used pool-device) before (row-id r))))
