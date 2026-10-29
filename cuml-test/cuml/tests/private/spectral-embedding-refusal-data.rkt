#lang racket/base

(require (only-in cuml/manifold spectral-embedding-fit-graph spectral-embedding-result?)
         (only-in racket/function const)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/probe probe-pool-used)
         (only-in "kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "refusal-rows.rkt" check-row row row-call row-id)
         (only-in "spectral-data.rkt"
                  clique-graph
                  graph->vectors
                  graph-edit
                  graph-map-vals
                  graph-upper
                  set-entry
                  unit-path
                  unit-rings
                  unit-star)
         (only-in "spectral-embedding-data.rkt"
                  embedded-graph
                  heavy-pair
                  stochastic-graph
                  with-isolated))

(provide (struct-out heavy-row)
         asymmetric-rows
         check-pool-kept
         check-then-valid
         eigen-refusal-rows
         eigen-rows
         fault
         graph-shape-rows
         heavy-rows
         path
         unit-weight-rows
         zero-text)

(struct heavy-row row (advised?))

(define pool-device 0)

(define ring-n 10)

(define ring (unit-rings 1 ring-n))

(define path (unit-path ring-n))

(define zero-text "embedding: column 0 is zero: the eigensolver found no eigenvector of this graph")

(define nan-text "embedding: row 0, column 0: the eigensolver returned nan")

(define solver-advice
  (string-append "; graphs whose Laplacian repeats eigenvalues (rings, grids, complete graphs, stars,"
                 " identical pieces) are beyond cuML's eigensolver"))

(define spread-advice "; these weights span orders of magnitude and need norm-laplacian")

(define asymmetric-advice "; the graph is not symmetric")

(define edgeless
  (string-append solver-advice "; samples without edges can displace the graph's eigenvectors"))

(define solver-text (string-append zero-text solver-advice))

(define heavy-text (string-append solver-text spread-advice))

(define past-sum (const (exact->inexact (expt 2 125))))

(define (inaccurate column [what ""] [advice solver-advice])
  (format "embedding: column ~a~a: the eigensolver's eigenvectors are inaccurate on this graph~a"
          column
          what
          advice))

(define (index-message name entry got)
  (format "~a: entry ~a: expected an index in [0, 10), got ~a" name entry got))

(define (fault g [options (hasheq)])
  (lambda () (embedded-graph g options)))

(define (fit-vectors rows cols vals [n ring-n])
  (spectral-embedding-fit-graph rows cols vals n))

(define (with-vectors f)
  (lambda ()
    (match-define (list rows cols vals) (graph->vectors ring))
    (f rows cols vals)))

(define (heavy-star w [g (unit-star 30)])
  (heavy-pair w #:graph g #:between '(0 1)))

(define split-ring
  (graph-edit (graph-map-vals ring (const 0.3))
              #:rows (lambda (rows) (append rows '(0)))
              #:cols (lambda (cols) (append cols '(1)))
              #:vals (lambda (vals) (append ((set-entry 0 0.1) vals) '(0.2)))))

(define (zero-row id g options)
  (row id solver-text (fault g options)))

(define (heavy id message g advised?)
  (heavy-row id message (fault g (hasheq 'norm-laplacian #f)) advised?))

(define (graph-shape-rows)
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

(define (zero-column-rows)
  (list (row "zero column" heavy-text (fault (heavy-pair (expt 2.0 24)) (hasheq 'norm-laplacian #f)))
        (zero-row "zero column, two rings" (unit-rings 2 30) (hasheq 'random-state 1))))

(define (unit-weight-rows)
  (list (zero-row "zero column, one ring" (unit-rings 1 30) (hasheq))
        (zero-row "zero column, star" (unit-star 30) (hasheq))
        (zero-row "zero column, unit-weight star" (unit-star 30) (hasheq 'norm-laplacian #f))))

(define (eigen-rows)
  (list (row "inaccurate, two rings" (inaccurate 0) (fault (unit-rings 2 30)))
        (row "inaccurate, two rings without norm-laplacian"
             (inaccurate 0)
             (fault (unit-rings 2 30) (hasheq 'norm-laplacian #f 'random-state 1)))
        (row "constant, one ring" (inaccurate 0 " is constant") (fault ring))
        (row "constant, one ring and an isolated sample"
             (inaccurate 0 " is constant" edgeless)
             (fault (with-isolated ring 1)))
        (row "constant, a ring whose (0, 1) is 0.1 + 0.2 and (1, 0) is 0.3"
             (inaccurate 0 " is constant")
             (fault split-ring))
        (row "constant, a path of 12 and 9 samples without edges"
             (inaccurate 0 " is constant" edgeless)
             (fault (with-isolated (unit-path 12) 9) (hasheq 'n-components 6)))
        (row "non-finite, K10 without norm-laplacian"
             (string-append nan-text solver-advice)
             (fault (clique-graph 10) (hasheq 'norm-laplacian #f 'random-state 3)))))

(define (asymmetric-rows)
  (list (row "asymmetric, upper triangle"
             (inaccurate 0 "" asymmetric-advice)
             (fault (graph-upper (stochastic-graph))))
        (row "asymmetric, upper triangle without norm-laplacian"
             (inaccurate 0 "" asymmetric-advice)
             (fault (graph-upper (stochastic-graph)) (hasheq 'norm-laplacian #f)))
        (row "asymmetric, upper path"
             (inaccurate 0 "" asymmetric-advice)
             (fault (graph-upper path) (hasheq 'random-state 1)))
        (row "asymmetric, upper path, the eigensolver returning nan"
             (string-append nan-text asymmetric-advice)
             (fault (graph-upper path)))))

(define (heavy-rows)
  (list
   (heavy "heavy, star with a pair at 2^7" solver-text (heavy-star 128.0) #f)
   (heavy "heavy, star with a pair at 2^8" heavy-text (heavy-star 256.0) #t)
   (heavy-row "heavy, star with a pair at 2^12 and norm-laplacian"
              solver-text
              (fault (heavy-star 4096.0))
              #f)
   (heavy "heavy, star with a pair at 2^7 twice" heavy-text (heavy-star 128.0 (heavy-star 128.0)) #t)
   (heavy "heavy, star with a pair summing to 2^8" heavy-text (heavy-star 255.0) #t)))

(define (eigen-refusal-rows)
  (append (zero-column-rows) (unit-weight-rows) (eigen-rows) (asymmetric-rows) (heavy-rows)))

(define (check-then-valid r)
  (check-row 'spectral-embedding-fit-graph r)
  (check-pred spectral-embedding-result? ((fault path)) (row-id r)))

(define (check-pool-kept r #:who [who 'spectral-embedding-fit-graph] #:call [thunk (row-call r)])
  (drain-finalizers!)
  (define before (probe-pool-used pool-device))
  (check-row who r thunk)
  (drain-finalizers!)
  (check-equal? (probe-pool-used pool-device) before (row-id r)))
