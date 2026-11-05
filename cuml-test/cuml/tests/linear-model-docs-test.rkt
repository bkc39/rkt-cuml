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
         (only-in racket/match match-define)
         (only-in rackunit check-equal?)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt"
                  oracle-ridge
                  [on-matrix matrix]
                  [on-vector vector]
                  ridge-solution))

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

(define (decimals k)
  (lambda (c) (real->decimal-string c k)))

(test-gpu "the ridge examples print what the exact oracle gives"
  (define rows '((1 1) (1 2) (2 2) (2 3)))
  (define ys '(6 8 9 11))
  (define X (matrix rows))
  (define y (vector ys))
  (define (oracle alpha)
    (match-define (ridge-solution _ coef intercept _ _ _ _) (oracle-ridge rows ys #f alpha #t))
    (values coef intercept))
  (define-values (c1 b1) (oracle 1))
  (define fitted-r (ridge-fit X y #:alpha 1.0))
  (check-equal? (~> fitted-r ridge-result-coef device-vector->list (map (decimals 6) _))
                (map (decimals 6) c1))
  (check-equal? (~> fitted-r (ridge-predict (matrix '((3 5) (2 5)))) device-vector->list)
                (for/list ([row (in-list '((3 5) (2 5)))])
                  (real->double-flonum (+ b1
                                          (for/sum ([x (in-list row)] [c (in-list c1)]) (* x c))))))
  (check-equal?
   (~> (ridge-fit X y #:alpha 0) ridge-result-coef device-vector->list (map (decimals 6) _))
   '("1.000000" "2.000000"))
  (define-values (c6 _b6) (oracle 1000000))
  (check-equal?
   (~> (ridge-fit X y #:alpha 1e6) ridge-result-coef device-vector->list (map (decimals 9) _))
   (map (decimals 9) c6))
  (check-cuml-error 'logic
                    "ridge-fit: alpha: expected a finite value at least 0, got -1"
                    (lambda () (ridge-fit X y #:alpha -1)))
  (check-cuml-error 'logic
                    "ridge-fit: solver: expected auto, eig or svd"
                    (lambda () (ridge-fit X y #:solver 'lsmr)))
  (define-values (ch bh) (oracle 1/2))
  (define mean (/ (apply + ys) (length ys)))
  (define residual
    (for/sum ([row (in-list rows)] [v (in-list ys)])
             (expt (- v bh (for/sum ([x (in-list row)] [c (in-list ch)]) (* x c))) 2)))
  (define total (for/sum ([v (in-list ys)]) (expt (- v mean) 2)))
  (check-equal? (~> (ridge #:alpha 0.5) (fit X y) (score X y) (real->decimal-string 6))
                (real->decimal-string (- 1 (/ residual total)) 6))
  (check-cuml-error 'logic "fit: ridge needs y" (lambda () (fit (ridge) X))))
