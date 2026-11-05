#lang racket/base

(require (only-in cuml fit linear-regression score)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
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
