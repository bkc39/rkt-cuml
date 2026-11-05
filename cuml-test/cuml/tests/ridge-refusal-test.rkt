#lang racket/base

(require (only-in cuml fit ridge)
         (only-in cuml/linear-model linear-regression-result ridge-fit ridge-params ridge-predict)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt"
                  check-law
                  dtypes
                  exact-fit
                  gap
                  lcase
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  oracle-ridge
                  [on-matrix matrix]
                  [on-vector vector]
                  ridge-solution)
         (only-in "private/linear-refusal-rows.rkt"
                  grid
                  overflow-X
                  overflow-y
                  shared-fit-rows
                  targets)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/ridge-data.rkt" ridge-case-fit solvers))

(define (fit-rows go)
  (define X (matrix grid))
  (define y (vector targets))
  (append (list (row "non-boolean fit-intercept before the X views"
                     "fit-intercept: expected #t or #f, got 'yes"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:fit-intercept 'yes #:alpha 'x)))
                (row "non-real alpha before the X views"
                     "alpha: expected a real number, got 'x"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:alpha 'x)))
                (row "X dtype before alpha"
                     "X: expected float32 or float64, got int32"
                     (lambda () (go (blank-matrix 5 3 'int32) y #:alpha -1)))
                (row "unknown solver"
                     "solver: expected auto, eig or svd"
                     (lambda () (go X y #:solver 'lsmr)))
                (row "unknown solver before alpha"
                     "solver: expected auto, eig or svd"
                     (lambda () (go X y #:solver 'lsmr #:alpha -1)))
                (row "negative alpha"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X y #:alpha -1)))
                (row "infinite alpha"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha +inf.0)))
                (row "nan alpha"
                     "alpha: expected a finite value at least 0, got nan"
                     (lambda () (go X y #:alpha +nan.0)))
                (row "alpha past the double range"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha (expt 10 400))))
                (row "alpha before a non-finite y"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X (vector '(1 2 3 4 +inf.0)) #:alpha -1)))
                (row "coefficient overflow"
                     "X: column 2: the coefficient exceeds the float32 range"
                     (lambda () (go (overflow-X) (overflow-y) #:alpha (expt 2 -250)))))
          (shared-fit-rows go)))

(define (estimator-fit X
                       y
                       #:sample-weight [w #f]
                       #:alpha [alpha 1.0]
                       #:solver [s 'auto]
                       #:fit-intercept [fi #t])
  (fit (ridge #:alpha alpha #:solver s #:fit-intercept fi) X y #:sample-weight w))

(test-gpu "ridge-fit refuses each row, the earlier fault first (Q1)"
  (for ([r (in-list (fit-rows ridge-fit))])
    (test-case (row-id r)
      (check-row "ridge-fit" r))))

(test-gpu "the estimator's fit names the generic"
  (for ([r (in-list (fit-rows estimator-fit))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "ridge-predict refuses a result of another kind, and the estimator a missing y"
  (check-exn #rx"^ridge-predict: contract violation\n  expected: ridge-result\\?"
             (lambda ()
               (ridge-predict (linear-regression-result (vector '(1 2 3)) 0.0) (matrix grid))))
  (check-cuml-error 'logic
                    "ridge-predict: X: expected 3 columns, got 2"
                    (lambda ()
                      (ridge-predict (ridge-fit (matrix grid) (vector targets)) (matrix '((1 2))))))
  (check-cuml-error 'logic "fit: ridge needs y" (lambda () (fit (ridge) (matrix grid))))
  (check-cuml-error 'logic
                    "ridge-params: ridge has no parameter algorithm"
                    (lambda () (ridge-params (hasheq 'algorithm 'eig)))))

(test-gpu "alpha 0, 0.0 and -0.0 are accepted; inf, nan and -1 are not (Q3)"
  (for ([alpha (in-list '(0 0.0 -0.0))])
    (check-true (list? (ridge-case-fit (normal-case 20 2 0) 'float64 #:alpha alpha))))
  (for ([alpha (in-list '(+inf.0 +nan.0 -1.0))])
    (check-cuml-error 'logic
                      #rx"^ridge-fit: alpha: expected a finite value at least 0, got "
                      (lambda () (ridge-case-fit (normal-case 20 2 0) 'float64 #:alpha alpha)))))

(define mixed-scales
  (let ([c (normal-case 60 4 11)])
    (lcase "columns at 2^-13, 1, 2^13, 1"
           (for/list ([row (in-list (lcase-X c))])
             (map * row (list (expt 2 -13) 1 (expt 2 13) 1)))
           (lcase-y c)
           #f)))

(test-gpu "float32 eig on columns of very different scales is refused; svd fits them (Q2)"
  (check-cuml-error 'logic
                    #rx"^ridge-fit: X: the eig solve is not a ridge solution "
                    (lambda () (ridge-case-fit mixed-scales 'float32 #:alpha 1024 #:solver 'eig)))
  (match-define (ridge-solution _ c* _ _ _ _ _)
    (oracle-ridge (lcase-X mixed-scales) (lcase-y mixed-scales) #f 1024 #t))
  (match-define (list coef _)
    (exact-fit (ridge-case-fit mixed-scales 'float32 #:alpha 1024 #:solver 'svd)))
  (check-law "svd gap" (gap coef c*) (* (expt 2 -10) (max-abs c*))))

(test-gpu "X, y and w are unchanged after every fit"
  (for* ([dt (in-list dtypes)]
         [s (in-list solvers)]
         [w (in-list (list #f '(1 0 2 3 1/2)))]
         [fi (in-list '(#t #f))])
    (define X (matrix grid dt))
    (define y (vector targets dt))
    (define W (and w (vector w dt)))
    (define (snapshot)
      (list (device-matrix->list* X) (device-vector->list y) (and W (device-vector->list W))))
    (define before (snapshot))
    (ridge-fit X y #:sample-weight W #:alpha 1/2 #:solver s #:fit-intercept fi)
    (test-case (format "~a ~a ~a ~a" dt s (and w #t) fi)
      (check-equal? (snapshot) before))))
