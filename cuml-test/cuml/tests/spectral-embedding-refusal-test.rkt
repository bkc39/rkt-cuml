#lang racket/base

(require (only-in cuml fit fit-transform spectral-embedding transform)
         (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-transform
                  spectral-embedding-params
                  spectral-embedding-result?)
         (only-in racket/flonum flsingle)
         (only-in racket/list make-list take)
         (only-in racket/match match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check-equal? check-exn check-pred check-true fail test-case)
         (only-in raft exn:fail:raft? list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe test-pools)
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
         (only-in "private/spectral-embedding-data.rkt" embedded polygons rows-matrix sign-ruled?)
         (only-in "private/spectral-embedding-refusal-data.rkt" check-pool-kept zero-text)
         (only-in "private/umap-data.rkt" far-blobs finite-rows? fixture-X fixture-rows tiny))

(struct X-case row (retry?))

(define small (hasheq 'n-neighbors 4))

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

(define zero-message (string-append zero-text "; deduplicate or centre X, or change n-neighbors"))

(define (halves)
  (rows->matrix (append (make-list 100 '(1.0 2.0)) (make-list 100 '(3.0 5.0))) 'float32))

(define (offset)
  (rows->matrix (line 64 1e9) 'float32))

(define (X-row id message X [options small] #:retry? [retry? #f])
  (X-case id message (cons X options) retry?))

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
   (X-row "s7" two-message (line-X 7) (hasheq 'n-neighbors 2) #:retry? #t)
   (X-row "s7 drop-first #f"
          "n-components: 2 components need at least 6 rows, got 5"
          (line-X 5)
          (hasheq 'n-neighbors 2 'drop-first #f))
   (X-row "s7 one component"
          "n-components: 1 component with drop-first needs at least 6 rows, got 5"
          (line-X 5)
          (hasheq 'n-components 1))
   (X-row "s9" "n-neighbors: expected at most the 8 rows, got 9" (tiny-X) (hasheq 'n-neighbors 9))
   (X-row "s10" nan-message (tiny-with nan-row) #:retry? #t)
   (X-row "s10 inf"
          "X: row 0, column 2: expected a finite value, got -inf"
          (tiny-with '([(0 2) -inf.0])))
   (X-row "s11" large-message (tiny-with large-row) #:retry? #t)
   (X-row "s12"
          "X: every row equals row 0; a spectral embedding needs two distinct rows"
          (rows->matrix (make-list 8 '(1.5 1.5 1.5)) 'float32)
          #:retry? #t)
   (X-row "s14 halves" zero-message (halves) (hasheq 'n-neighbors 10) #:retry? #t)
   (X-row "s14 offset" zero-message (offset) (hasheq 'n-neighbors 10) #:retry? #t)
   (X-row "s14 every row a neighbour" zero-message (line-X 50) (hasheq 'n-neighbors 50))
   (X-row "s14 identical pieces"
          zero-message
          (rows-matrix (polygons 2 30))
          (hasheq 'n-neighbors 3 'random-state 1))
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

(define (fit-thunk r)
  (match-define (cons X options) (row-call r))
  (lambda () (apply-options spectral-embedding-fit options X)))

(define contract-options
  (list (hasheq 'n-neighbors 2.5)
        (hasheq 'n-components 'two)
        (hasheq 'random-state -1)
        (hasheq 'random-state (expt 2 64))))

(define least-rows
  '((1 #f 4 "1 component needs at least 4 rows, got 3")
    (1 #t 6 "1 component with drop-first needs at least 6 rows, got 5")
    (2 #t 8 "2 components with drop-first need at least 8 rows, got 7")
    (2 #f 6 "2 components need at least 6 rows, got 5")
    (8 #t 20 "8 components with drop-first need at least 20 rows, got 19")
    (19 #t 42 "19 components with drop-first need at least 42 rows, got 41")))

(test-gpu "L13: the fits refuse each bad X or parameter with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (X-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (match-define (cons X options) (row-call r))
      (check-row who r (lambda () (call X options)))))
  (for ([options (in-list contract-options)])
    (check-exn exn:fail:contract?
               (lambda () (apply-options spectral-embedding-fit options (tiny-X))))))

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

(test-gpu "L13: s7: the least rows fit for every probed (n-components, drop-first)"
  (for ([entry (in-list least-rows)])
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
    (check-pool-kept
     (row (symbol->string who) zero-message (lambda () (call (halves) (hasheq 'n-neighbors 10))))
     #:who who)))

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
        #:when (X-case-retry? r))
    (check-row 'spectral-embedding-fit r (fit-thunk r))
    (check-pred spectral-embedding-result?
                (spectral-embedding-fit (valid-X) #:n-neighbors 10)
                (row-id r))))

(test-pools "L13: a refusal leaves the RMM pool where it was"
  (for ([r (in-list (X-rows))])
    (check-pool-kept r #:who 'spectral-embedding-fit #:call (fit-thunk r))))
