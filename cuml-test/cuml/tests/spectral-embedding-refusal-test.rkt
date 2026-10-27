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
         (only-in "private/spectral-embedding-data.rkt" sign-ruled? embedded)
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

(define (lanczos-message k drop least got)
  (format "n-components: ~a component~a~a need~a at least ~a rows, got ~a"
          k
          (if (= k 1) "" "s")
          (if drop " with drop-first" "")
          (if (= k 1) "s" "")
          least
          got))

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
   (X-row "s7" (lanczos-message 2 #t 8 7) (line-X 7) (hasheq 'n-neighbors 2))
   (X-row "s7 drop-first #f"
          (lanczos-message 2 #f 6 5)
          (line-X 5)
          (hasheq 'n-neighbors 2 'drop-first #f))
   (X-row "s7 one component" (lanczos-message 1 #t 6 5) (line-X 5) (hasheq 'n-components 1))
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
   (X-row "s7 before s9" (lanczos-message 2 #t 8 7) (line-X 7) (hasheq 'n-neighbors 9))
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
                    (lambda () (fit (spectral-embedding #:affinity 'precomputed) 'X))))

(define (fit-row r)
  (match-define (cons X options) (row-call r))
  (check-row 'spectral-embedding-fit r (lambda () (apply-options spectral-embedding-fit options X))))

(test-gpu "L13: s7: the least rows fit for every probed (n-components, drop-first)"
  (for ([entry (in-list '((1 #t) (2 #t) (2 #f) (8 #t) (19 #t)))])
    (match-define (list k drop) entry)
    (define least (* 2 (+ k (if drop 1 0) 1)))
    (test-case (format "k ~a drop ~a" k drop)
      (define options
        (hasheq 'n-components k 'drop-first drop 'n-neighbors (add1 (quotient least 2))))
      (check-cuml-error
       'logic
       (format "spectral-embedding-fit: ~a" (lanczos-message k drop least (sub1 least)))
       (lambda () (apply-options spectral-embedding-fit options (line-X (sub1 least)))))
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
