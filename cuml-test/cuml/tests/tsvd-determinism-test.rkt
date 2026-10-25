#lang racket/base

(require (only-in cuml/decomposition tsvd-fit-transform)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix)
         (only-in "private/tsvd-data.rkt" normal-rows-of svd-components svd-ratio svd-singular svd-variance tsvd-of))

(define (outcome X solver)
  (define-values (result T) (tsvd-fit-transform X #:n-components 8 #:solver solver))
  (define fitted (tsvd-of result))
  (list (svd-components fitted)
        (svd-variance fitted)
        (svd-ratio fitted)
        (svd-singular fitted)
        (device-matrix->list* T)))

(test-gpu "L12: five fits of one X give the same bits, in both dtypes and under both solvers"
  (for* ([n (in-list '(4096 20000))]
         [rows (in-value (normal-rows-of (+ n 1) n 64))]
         [dt (in-list '(float32 float64))]
         [X (in-value (rows->matrix rows dt))]
         [solver (in-list '(full jacobi))])
    (test-case (format "~ax64 ~a ~a" n dt solver)
      (define first-outcome (outcome X solver))
      (for ([_ (in-range 4)])
        (check-true (bits-equal? (outcome X solver) first-outcome))))))
