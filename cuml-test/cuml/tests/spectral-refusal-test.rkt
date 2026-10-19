#lang racket/base

(require (only-in cuml/cluster spectral-clustering-fit spectral-clustering-fit-graph)
         (only-in racket/function curry)
         (only-in racket/list list-set make-list)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-exn fail test-case)
         (only-in raft
                  device-resources
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  list->device-vector
                  with-device-resources)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/spectral-data.rkt"
                  graph
                  graph-cols
                  graph-labels
                  graph-rows
                  graph-vals
                  labels))

(struct row (id message call))

(define (grid dt [n 10])
  (list*->device-matrix (for/list ([i (in-range n)])
                          (list (add1 i) (* 2 i)))
                        #:dtype dt))

(define (with-row bad)
  (list*->device-matrix
   (append '((1 1) (2 3)) (list bad) '((4 4) (5 6)) (list '(+nan.0 0)) '((7 7) (8 9) (9 1) (3 3)))
   #:dtype 'float32))

(define (norm-message got)
  (format "X: row 2: expected a squared norm of at most 4.253529586511731e+37, got ~a" got))

(define tol-most "eigen-tol: expected at most 3.4028234663852886e+38, got ")

(define (dense-rows)
  (define X (grid 'float32))
  (define fit spectral-clustering-fit)
  (list
   (row "D0 float64" "X: expected float32, got float64" (lambda () (fit (grid 'float64))))
   (row "D0 int32" "X: expected float32, got int32" (lambda () (fit (blank-matrix 10 2 'int32))))
   (row "D1 rank" "X: expected rank 2, got rank 1" (lambda () (fit (blank-vector 10 'float32))))
   (row "D1 layout"
        "X: expected row-major, got col-major 10x2"
        (lambda () (fit (blank-matrix 10 2 'float32 #:layout 'col-major))))
   (row "D2" "X: expected at least 4 rows, got 3" (lambda () (fit (grid 'float32 3))))
   (row "D2 empty" "X: expected at least 4 rows, got 0" (lambda () (fit (blank-matrix 0 2 'float32))))
   (row "D3" "X: expected at least 1 column, got 0" (lambda () (fit (blank-matrix 10 0 'float32))))
   (row "D5" "n-clusters: expected at least 1, got 0" (lambda () (fit X #:n-clusters 0)))
   (row "D5 negative" "n-clusters: expected at least 1, got -2" (lambda () (fit X #:n-clusters -2)))
   (row "D6"
        "X: 5 samples are fewer than the 6 clusters"
        (lambda () (fit (grid 'float32 5) #:n-clusters 6)))
   (row "D7 zero" "n-components: expected at least 1, got 0" (lambda () (fit X #:n-components 0)))
   (row "D7 most"
        "n-components: expected at most 3 for 10 rows, got 4"
        (lambda () (fit X #:n-clusters 2 #:n-components 4)))
   (row "D8" "n-neighbors: expected at least 2, got 1" (lambda () (fit X #:n-neighbors 1)))
   (row "D8 negative" "n-neighbors: expected at least 2, got -4" (lambda () (fit X #:n-neighbors -4)))
   (row "D9" "n-init: expected at least 1, got 0" (lambda () (fit X #:n-init 0)))
   (row "D10 negative" "eigen-tol: expected at least 0, got -1" (lambda () (fit X #:eigen-tol -1)))
   (row "D10 nan" "eigen-tol: expected at least 0, got nan" (lambda () (fit X #:eigen-tol +nan.0)))
   (row "D10 inf" (string-append tol-most "inf") (lambda () (fit X #:eigen-tol +inf.0)))
   (row "D10 1e39" (string-append tol-most "1e+39") (lambda () (fit X #:eigen-tol 1e39)))
   (row "D12 nan" (norm-message "nan") (lambda () (fit (with-row '(+nan.0 1)))))
   (row "D12 inf" (norm-message "inf") (lambda () (fit (with-row '(+inf.0 0)))))
   (row "D12 -inf" (norm-message "inf") (lambda () (fit (with-row '(0 -inf.0)))))
   (row "D12 1e20" (norm-message "1.0000000400817551e+40") (lambda () (fit (with-row '(1e20 0)))))
   (row "D12 2^63"
        (norm-message "8.507059173023462e+37")
        (lambda () (fit (with-row (list (expt 2 63) 0)))))
   (row "D0 before D2" "X: expected float32, got float64" (lambda () (fit (grid 'float64 3))))
   (row "D2 before D3"
        "X: expected at least 4 rows, got 3"
        (lambda () (fit (blank-matrix 3 0 'float32))))
   (row "D5 before D7"
        "n-clusters: expected at least 1, got 0"
        (lambda () (fit X #:n-clusters 0 #:n-components 0)))
   (row "D7 before D8"
        "n-components: expected at least 1, got 0"
        (lambda () (fit X #:n-components 0 #:n-neighbors 1)))
   (row "D8 before D9"
        "n-neighbors: expected at least 2, got 1"
        (lambda () (fit X #:n-neighbors 1 #:n-init 0)))
   (row "D9 before D10"
        "n-init: expected at least 1, got 0"
        (lambda () (fit X #:n-init 0 #:eigen-tol -1)))
   (row "D10 before D12"
        "eigen-tol: expected at least 0, got -1"
        (lambda () (fit (with-row '(+nan.0 1)) #:eigen-tol -1)))))

(define ring-n 10)

(define ring
  (let ([pairs (for*/list ([i (in-range ring-n)]
                           [e (in-list (list (cons i (modulo (add1 i) ring-n))
                                             (cons (modulo (add1 i) ring-n) i)))])
                 e)])
    (graph (map car pairs) (map cdr pairs) (make-list (length pairs) 1.0) ring-n)))

(define (edited #:rows [rows values] #:cols [cols values] #:vals [vals values])
  (graph (rows (graph-rows ring)) (cols (graph-cols ring)) (vals (graph-vals ring)) ring-n))

(define ((at i v) xs)
  (list-set xs i v))

(define (vectors g)
  (list (list->device-vector (graph-rows g) #:dtype 'int32)
        (list->device-vector (graph-cols g) #:dtype 'int32)
        (list->device-vector (graph-vals g) #:dtype 'float32)))

(define (fit-vectors rows cols vals [n ring-n])
  (spectral-clustering-fit-graph rows cols vals n))

(define (index-message name entry got)
  (format "~a: entry ~a: expected an index in [0, 10), got ~a" name entry got))

(define (weight-message entry got)
  (format "vals: entry ~a: expected a finite weight of at least 0, got ~a" entry got))

(define self-loop-message "rows: entry 12: self-loop at sample 5; leave the diagonal out")

(define (graph-rows-of)
  (define (fault g)
    (lambda () (graph-labels g #:n-clusters 2)))
  (define self-loop (edited #:rows (at 12 5) #:cols (at 12 5)))
  (list
   (row "G0 rows float32"
        "rows: expected int32, got float32"
        (lambda ()
          (match (vectors ring)
            [(list _ cols vals) (fit-vectors (blank-vector 20 'float32) cols vals)])))
   (row "G0 rows int64"
        "rows: expected int32, got int64"
        (lambda ()
          (match (vectors ring)
            [(list _ cols vals) (fit-vectors (blank-vector 20 'int64) cols vals)])))
   (row "G0 vals float64"
        "vals: expected float32, got float64"
        (lambda ()
          (match (vectors ring)
            [(list rows cols _) (fit-vectors rows cols (blank-vector 20 'float64))])))
   (row "G1 rows"
        "rows: expected rank 1, got rank 2"
        (lambda ()
          (match (vectors ring)
            [(list _ cols vals) (fit-vectors (blank-matrix 20 1 'int32) cols vals)])))
   (row "G2 cols"
        "cols: expected 20 entries, as rows has, got 19"
        (lambda ()
          (match (vectors ring)
            [(list rows _ vals) (fit-vectors rows (blank-vector 19 'int32) vals)])))
   (row "G2 vals"
        "vals: expected 20 entries, as rows has, got 21"
        (lambda ()
          (match (vectors ring)
            [(list rows cols _) (fit-vectors rows cols (blank-vector 21 'float32))])))
   (row "G3 few"
        "n-samples: expected at least 4, got 3"
        (lambda () (apply fit-vectors (append (vectors ring) '(3)))))
   (row "G3 many"
        "n-samples: expected at most 2147483647, got 2147483648"
        (lambda () (apply fit-vectors (append (vectors ring) (list (expt 2 31))))))
   (row "G5 zero"
        "n-clusters: expected at least 1, got 0"
        (lambda () (graph-labels ring #:n-clusters 0)))
   (row "G5 most"
        "n-samples: 10 samples are fewer than the 11 clusters"
        (lambda () (graph-labels ring #:n-clusters 11)))
   (row "G6"
        "n-components: expected at most 3 for 10 samples, got 4"
        (lambda () (graph-labels ring #:n-clusters 2 #:n-components 4)))
   (row "G7" "n-init: expected at least 1, got 0" (lambda () (graph-labels ring #:n-init 0)))
   (row "G8" "eigen-tol: expected at least 0, got -1" (lambda () (graph-labels ring #:eigen-tol -1)))
   (row "G9 rows" (index-message "rows" 7 10) (fault (edited #:rows (at 7 10))))
   (row "G9 rows -1" (index-message "rows" 7 -1) (fault (edited #:rows (at 7 -1))))
   (row "G9 cols" (index-message "cols" 7 10) (fault (edited #:cols (at 7 10))))
   (row "G9 cols -1" (index-message "cols" 7 -1) (fault (edited #:cols (at 7 -1))))
   (row "G9 10^8" (index-message "rows" 7 100000000) (fault (edited #:rows (at 7 100000000))))
   (row "G10" self-loop-message (fault self-loop))
   (row "G11 negative" (weight-message 3 -5) (fault (edited #:vals (at 3 -5.0))))
   (row "G11 nan" (weight-message 3 "nan") (fault (edited #:vals (at 3 +nan.0))))
   (row "G11 inf" (weight-message 3 "inf") (fault (edited #:vals (at 3 +inf.0))))
   (row "G12 zeros"
        "vals: expected at least one positive weight, got none"
        (fault (edited #:vals (lambda (vs) (map (lambda (_) 0.0) vs)))))
   (row "G12 empty"
        "vals: expected at least one positive weight, got none"
        (lambda ()
          (fit-vectors (blank-vector 0 'int32) (blank-vector 0 'int32) (blank-vector 0 'float32))))
   (row "G2 before G9"
        "vals: expected 20 entries, as rows has, got 19"
        (lambda ()
          (match (vectors (edited #:rows (at 7 10)))
            [(list rows cols _) (fit-vectors rows cols (blank-vector 19 'float32))])))
   (row "G9 before G10 at one entry"
        (index-message "rows" 7 10)
        (fault (edited #:rows (at 7 10) #:cols (at 7 10))))
   (row "G9 before G11 at one entry"
        (index-message "cols" 7 -1)
        (fault (edited #:cols (at 7 -1) #:vals (at 7 -5.0))))
   (row "G10 before G11 at one entry"
        "rows: entry 7: self-loop at sample 4; leave the diagonal out"
        (fault (edited #:rows (at 7 4) #:cols (at 7 4) #:vals (at 7 +nan.0))))
   (row "the earliest entry wins"
        (weight-message 3 -5)
        (fault (edited #:rows (at 7 10) #:vals (at 3 -5.0))))))

(test-gpu "spectral-clustering-fit refuses each bad argument with its message"
  (for ([r (in-list (dense-rows))])
    (test-case (row-id r)
      (check-cuml-error 'logic (format "spectral-clustering-fit: ~a" (row-message r)) (row-call r)))))

(test-gpu "spectral-clustering-fit-graph refuses each bad argument with its message"
  (for ([r (in-list (graph-rows-of))])
    (test-case (row-id r)
      (check-cuml-error 'logic
                        (format "spectral-clustering-fit-graph: ~a" (row-message r))
                        (row-call r)))))

(test-gpu "R3: the graph-size limit is refused before the row scan"
  (check-cuml-error 'logic
                    (string-append "spectral-clustering-fit: X: 102261127 rows with 10 neighbours"
                                   " need 2147483667 graph entries, more than 2147483647")
                    (lambda () (spectral-clustering-fit (blank-matrix 102261127 1 'float32)))))

(test-gpu "R2: rows at the squared-norm boundary, and subnormal rows, fit"
  (define boundary
    (list*->device-matrix
     (append
      (list '(0 0) (list (expt 2 62) (expt 2 62)) (list (- (expt 2 62)) (- (expt 2 62))) '(1e-45 0))
      (for/list ([i (in-range 6)])
        (list i (* i i))))
     #:dtype 'float32))
  (check-equal? (length (labels boundary #:n-clusters 3)) 10))

(test-gpu "R5: values that do not fit the native types raise exn:fail:contract"
  (define X (grid 'float32))
  (define fit (curry spectral-clustering-fit X))
  (check-exn exn:fail:contract? (lambda () (fit #:n-init (expt 2 31))))
  (check-exn exn:fail:contract? (lambda () (fit #:random-state -1)))
  (check-exn exn:fail:contract? (lambda () (fit #:random-state (expt 2 64))))
  (check-exn exn:fail:contract? (lambda () (fit #:n-clusters 2.5)))
  (check-exn exn:fail:contract? (lambda () (fit #:n-components 1.5)))
  (check-exn exn:fail:contract? (lambda () (fit #:eigen-tol 'fast)))
  (check-exn exn:fail:contract? (lambda () (graph-labels ring #:n-init (expt 2 31))))
  (check-exn exn:fail:contract? (lambda () (graph-labels ring #:random-state -1))))

(test-gpu "R4: released resources are refused under the procedure's own name"
  (define released (with-device-resources ([r (device-resources)]) r))
  (for ([call (in-list (list (lambda ()
                               (spectral-clustering-fit (grid 'float32) #:resources released))
                             (lambda () (graph-labels ring #:resources released))))]
        [name (in-list '(spectral-clustering-fit spectral-clustering-fit-graph))])
    (match (raised call)
      [(? exn:fail:raft? e)
       (check-equal? (exn:fail:raft-kind e) 'logic)
       (check-equal? (exn-message e)
                     (format "~a: the device resources on device 0 were released" name))]
      [other (fail (format "expected an exn:fail:raft, got ~e" other))])))
