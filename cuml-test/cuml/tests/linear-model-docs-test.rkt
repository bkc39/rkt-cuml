#lang racket/base

(require (only-in cuml fit linear-regression ridge score)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept
                  ridge-fit
                  ridge-predict
                  ridge-result-coef)
         (only-in rackunit check-equal?)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" [on-matrix matrix] [on-vector vector]))

(test-gpu "the linear-regression examples print what the manual shows"
  (define X (matrix '((1 1) (1 2) (2 2) (2 3))))
  (define y (vector '(6 8 9 11)))
  (define fitted (linear-regression-fit X y))
  (check-equal? (~> fitted
                    linear-regression-result-coef
                    device-vector->list
                    (map (lambda (c) (real->decimal-string c 6)) _))
                '("1.000000" "2.000000"))
  (check-equal? (~> fitted (linear-regression-predict (matrix '((3 5) (2 5)))) device-vector->list)
                '(16.0 15.0))
  (check-equal? (~> (linear-regression-fit (matrix '((0 1) (1 1) (2 1) (3 1))) (vector '(1 3 5 7)))
                    linear-regression-result-coef
                    device-vector->list)
                '(2.0 0.0))
  (check-equal? (~> (linear-regression-fit X
                                           (vector '(6 8 10 11))
                                           #:sample-weight (vector '(1 2 3 4))
                                           #:algorithm 'qr)
                    linear-regression-result-intercept
                    (real->decimal-string 6))
                "3.120000")
  (check-equal? (~> (linear-regression) (fit X y) (score X y) (real->decimal-string 6)) "1.000000")
  (check-cuml-error 'logic
                    "linear-regression-fit: y: expected 4 elements, got 3"
                    (lambda () (linear-regression-fit X (vector '(1 2 3)))))
  (check-cuml-error 'logic
                    "linear-regression-predict: X: expected 2 columns, got 3"
                    (lambda () (linear-regression-predict fitted (matrix '((1 2 3))))))
  (check-cuml-error 'logic "fit: linear-regression needs y" (lambda () (fit (linear-regression) X))))

(define (decimals k cs)
  (map (lambda (c) (real->decimal-string c k)) cs))

(test-gpu "the ridge examples print what the manual shows"
  (define X (matrix '((1 1) (1 2) (2 2) (2 3))))
  (define y (vector '(6 8 9 11)))
  (define fitted (ridge-fit X y #:alpha 1.0))
  (check-equal? (~> fitted ridge-result-coef device-vector->list (decimals 6 _))
                '("0.800000" "1.400000"))
  (check-equal? (~> fitted (ridge-predict (matrix '((3 5) (2 5)))) device-vector->list) '(13.9 13.1))
  (check-equal? (~> (ridge-fit X y #:alpha 0) ridge-result-coef device-vector->list (decimals 6 _))
                '("1.000000" "2.000000"))
  (check-equal? (~> (ridge-fit X y #:alpha 1e6) ridge-result-coef device-vector->list (decimals 9 _))
                '("0.000003000" "0.000005000"))
  (check-cuml-error 'logic
                    "ridge-fit: alpha: expected a finite value at least 0, got -1"
                    (lambda () (ridge-fit X y #:alpha -1)))
  (check-cuml-error 'logic
                    "ridge-fit: solver: expected auto, eig or svd"
                    (lambda () (ridge-fit X y #:solver 'lsmr)))
  (check-equal? (~> (ridge #:alpha 0.5) (fit X y) (score X y) (real->decimal-string 6)) "0.973935")
  (check-cuml-error 'logic "fit: ridge needs y" (lambda () (fit (ridge) X))))
