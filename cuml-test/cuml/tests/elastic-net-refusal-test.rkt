#lang racket/base

(require (only-in cuml elastic-net fit lasso)
         (only-in cuml/linear-model
                  elastic-net-fit
                  elastic-net-params
                  elastic-net-predict
                  lasso-fit
                  lasso-predict
                  lasso-result
                  linear-regression-result)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/elastic-net-data.rkt"
                  check-s2
                  correlated-case
                  en-case-fit
                  law-alpha
                  norm2)
         (only-in "private/elastic-net-oracle.rkt" en-solution oracle-elastic-net problem-of)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linear-data.rkt"
                  check-law
                  dtypes
                  lcase
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  [on-matrix matrix]
                  [on-vector vector]
                  scaled-case
                  scaled-columns
                  unit-roundoff)
         (only-in "private/linear-refusal-rows.rkt" grid shared-fit-rows targets)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define (fit-rows go #:l1-ratio? [l1-ratio? #t])
  (define X (matrix grid))
  (define y (vector targets))
  (define bad-X (blank-matrix 5 3 'int32))
  (append (list (row "non-boolean fit-intercept before every other fault"
                     "fit-intercept: expected #t or #f, got 'yes"
                     (lambda () (go bad-X y #:fit-intercept 'yes #:alpha 'x #:tol 'x)))
                (row "non-real alpha before l1-ratio"
                     "alpha: expected a real number, got 'x"
                     (lambda () (go bad-X y #:alpha 'x #:max-iter 0 #:tol 'x))))
          (if l1-ratio?
              (list (row "non-real l1-ratio before max-iter"
                         "l1-ratio: expected a real number, got 'x"
                         (lambda () (go bad-X y #:l1-ratio 'x #:max-iter 0))))
              '())
          (list (row "max-iter 0 before tol"
                     "max-iter: expected an exact integer from 1 to 2147483647, got 0"
                     (lambda () (go bad-X y #:max-iter 0 #:tol 'x)))
                (row "max-iter past int32"
                     "max-iter: expected an exact integer from 1 to 2147483647, got 2147483648"
                     (lambda () (go X y #:max-iter 2147483648)))
                (row "non-integer max-iter"
                     "max-iter: expected an exact integer from 1 to 2147483647, got 1.5"
                     (lambda () (go X y #:max-iter 1.5)))
                (row "negative max-iter"
                     "max-iter: expected an exact integer from 1 to 2147483647, got -1"
                     (lambda () (go X y #:max-iter -1)))
                (row "non-real tol before the X views"
                     "tol: expected a real number, got 'x"
                     (lambda () (go bad-X y #:tol 'x)))
                (row "X dtype before selection"
                     "X: expected float32 or float64, got int32"
                     (lambda () (go bad-X y #:selection 'shuffled)))
                (row "unknown selection before alpha"
                     "selection: expected cyclic or random"
                     (lambda () (go X y #:selection 'shuffled #:alpha -1)))
                (row "negative alpha before l1-ratio"
                     "alpha: expected a finite value at least 0, got -1"
                     (lambda () (go X y #:alpha -1 #:tol -1)))
                (row "infinite alpha"
                     "alpha: expected a finite value at least 0, got inf"
                     (lambda () (go X y #:alpha +inf.0)))
                (row "nan alpha"
                     "alpha: expected a finite value at least 0, got nan"
                     (lambda () (go X y #:alpha +nan.0)))
                (row "negative tol"
                     "tol: expected a finite value of at least 0, got -1"
                     (lambda () (go X y #:tol -1)))
                (row "nan tol"
                     "tol: expected a finite value of at least 0, got nan"
                     (lambda () (go X y #:tol +nan.0)))
                (row "infinite tol"
                     "tol: expected a finite value of at least 0, got inf"
                     (lambda () (go X y #:tol +inf.0)))
                (row "tol before a non-finite X"
                     "tol: expected a finite value of at least 0, got -1"
                     (lambda () (go (matrix '((1 2) (+nan.0 1) (2 2))) (vector '(1 2 3)) #:tol -1))))
          (if l1-ratio?
              (list (row "l1-ratio below 0"
                         "l1-ratio: expected a finite value from 0 to 1, got -0.5"
                         (lambda () (go X y #:l1-ratio -0.5)))
                    (row "l1-ratio above 1"
                         "l1-ratio: expected a finite value from 0 to 1, got 1.5"
                         (lambda () (go X y #:l1-ratio 1.5)))
                    (row "nan l1-ratio"
                         "l1-ratio: expected a finite value from 0 to 1, got nan"
                         (lambda () (go X y #:l1-ratio +nan.0))))
              '())
          (filter (lambda (r) (not (equal? (row-id r) "intercept overflow"))) (shared-fit-rows go))))

(define (estimator-fit make)
  (lambda (X
           y
           #:sample-weight [w #f]
           #:alpha [alpha 1.0]
           #:l1-ratio [r 0.5]
           #:fit-intercept [fi #t]
           #:max-iter [k 1000]
           #:tol [tol 1e-3]
           #:selection [s 'cyclic])
    (fit (make #:alpha alpha #:l1-ratio r #:fit-intercept fi #:max-iter k #:tol tol #:selection s)
         X
         y
         #:sample-weight w)))

(define (lasso-go X
                  y
                  #:sample-weight [w #f]
                  #:alpha [alpha 1.0]
                  #:fit-intercept [fi #t]
                  #:max-iter [k 1000]
                  #:tol [tol 1e-3]
                  #:selection [s 'cyclic])
  (lasso-fit X
             y
             #:sample-weight w
             #:alpha alpha
             #:fit-intercept fi
             #:max-iter k
             #:tol tol
             #:selection s))

(test-gpu "elastic-net-fit refuses each row, the earlier fault first (Q1)"
  (for ([r (in-list (fit-rows elastic-net-fit))])
    (test-case (row-id r)
      (check-row "elastic-net-fit" r))))

(test-gpu "lasso-fit refuses each row but l1-ratio's, the earlier fault first (Q1)"
  (for ([r (in-list (fit-rows lasso-go #:l1-ratio? #f))])
    (test-case (row-id r)
      (check-row "lasso-fit" r))))

(test-gpu "the estimators' fit names the generic"
  (for ([r (in-list (fit-rows (estimator-fit elastic-net)))])
    (test-case (row-id r)
      (check-row "fit" r))))

(test-gpu "predict refuses a result of another kind, the estimators a missing y"
  (define X (matrix grid))
  (check-exn #rx"^elastic-net-predict: contract violation\n  expected: elastic-net-result\\?"
             (lambda () (elastic-net-predict (lasso-result (vector '(1 2 3)) 0.0 1) X)))
  (check-exn #rx"^lasso-predict: contract violation\n  expected: lasso-result\\?"
             (lambda () (lasso-predict (linear-regression-result (vector '(1 2 3)) 0.0) X)))
  (check-cuml-error 'logic
                    "lasso-predict: X: expected 3 columns, got 2"
                    (lambda () (lasso-predict (lasso-fit X (vector targets)) (matrix '((1 2))))))
  (check-cuml-error 'logic "fit: lasso needs y" (lambda () (fit (lasso) X)))
  (check-cuml-error 'logic "fit: elastic-net needs y" (lambda () (fit (elastic-net) X)))
  (check-cuml-error 'logic
                    "elastic-net-params: elastic-net has no parameter solver"
                    (lambda () (elastic-net-params (hasheq 'solver 'cd)))))

(define docs (lcase "docs" '((1 1) (1 2) (2 2) (2 3)) '(6 8 9 11) #f))

(test-gpu "an absolute stop short of the relative test is refused, and y scaled up fits (Q2)"
  (define wide (scaled-case docs #:X 100000))
  (define c (scaled-case (correlated-case 500 8 0.9 21) #:X 100000))
  (define p (problem-of (lcase-X c) (lcase-y c) #f #t))
  (define alpha (law-alpha p 0.01 1))
  (for ([dt (in-list dtypes)])
    (check-cuml-error
     'logic
     #rx"^elastic-net-fit: X: coordinate descent stopped after 1 epochs with every coefficient under tol 0.001 and KKT residual "
     (lambda () (en-case-fit wide dt #:alpha 1e4 #:l1-ratio 1)))
    (check-cuml-error 'logic
                      #rx"; tol is absolute in the coefficients' units, so scale y up or X down$"
                      (lambda () (en-case-fit c dt #:alpha alpha #:l1-ratio 1)))
    (define up (scaled-case c #:y (expt 2 17)))
    (define p-up (problem-of (lcase-X up) (lcase-y up) #f #t))
    (define alpha-up (law-alpha p-up 0.01 1))
    (match-define (list coef _ k) (en-case-fit up dt #:alpha alpha-up #:l1-ratio 1))
    (match-define (struct* en-solution ([coef c*] [mu mu] [sigma sigma]))
      (oracle-elastic-net p-up alpha-up 1))
    (check-true (< k 1000))
    (check-law "S2"
               (norm2 (map (lambda (a e) (- (inexact->exact a) e)) coef c*))
               (+ (/ (* 1e-3 (max-abs coef) (apply + sigma) (norm2 sigma)) mu)
                  (* 64 (unit-roundoff dt) (norm2 (map exact->inexact c*))))))
  (check-cuml-error 'logic
                    #rx"^elastic-net-fit: X: coordinate descent stopped after "
                    (lambda ()
                      (en-case-fit (scaled-case docs #:X (expt 10 20)) 'float32 #:alpha 1e-3))))

(test-gpu "hyperparameter edges: nan, inf and out-of-range values refused, tol 0 and alpha 0 run (Q3)"
  (define c (correlated-case 20 2 0 22))
  (for ([alpha (in-list '(+nan.0 +inf.0 -1.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: alpha: expected a finite value at least 0, got "
                      (lambda () (en-case-fit c 'float64 #:alpha alpha))))
  (for ([r (in-list '(-0.5 1.5 +nan.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: l1-ratio: expected a finite value from 0 to 1, got "
                      (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:l1-ratio r))))
  (for ([tol (in-list '(-1 +nan.0 +inf.0))])
    (check-cuml-error 'logic
                      #rx"^elastic-net-fit: tol: expected a finite value of at least 0, got "
                      (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:tol tol))))
  (for ([k (in-list (list 0 -1 (expt 2 31) 1.5))])
    (check-cuml-error
     'logic
     #rx"^elastic-net-fit: max-iter: expected an exact integer from 1 to 2147483647, got "
     (lambda () (en-case-fit c 'float64 #:alpha 0.1 #:max-iter k))))
  (check-equal? (caddr (en-case-fit c 'float64 #:alpha 0.1 #:tol 0 #:max-iter 9)) 9)
  (check-true (> (caddr (en-case-fit c 'float64 #:alpha 0)) 1)))

(test-gpu "float32 columns 2^120 apart are refused as too narrow; 2^60 apart fit (Q4)"
  (define c (correlated-case 100 2 0 23))
  (check-cuml-error
   'logic
   "elastic-net-fit: X: column 0 is too narrow beside the other columns and y for float32 coordinate descent"
   (lambda ()
     (en-case-fit (scaled-columns c (list (expt 2 -60) (expt 2 60)) 1) 'float32 #:alpha 1e-3)))
  (check-equal? (check-s2 (scaled-columns c (list (expt 2 -30) (expt 2 30)) 1) 'float32 1e-3 0.5 1e-3)
                "S2"))

(test-gpu "an alpha whose row multiple overflows float32 is refused with ridge's message (Q5)"
  (check-cuml-error
   'logic
   #rx"^elastic-net-fit: alpha: expected at most .* for this X in float32, got 1e\\+38$"
   (lambda () (en-case-fit (correlated-case 20 2 0 24) 'float32 #:alpha 1e38))))

(define stop-message
  #rx"^elastic-net-fit: X: coordinate descent stopped after 1 epochs with every coefficient under tol ")

(test-gpu "under a heavy L2 penalty, an absolute stop short of tol is still refused (L3-m)"
  (define c (scaled-case (correlated-case 200 7 0.9 31) #:y (expt 2 -10)))
  (for ([dt (in-list dtypes)])
    (check-cuml-error 'logic
                      stop-message
                      (lambda () (en-case-fit c dt #:alpha 1e4 #:l1-ratio 0 #:tol 1e-5)))))

(define tall-rows (expt 2 17))

(define nearly-orthogonal
  (let ([X (for/list ([i (in-range tall-rows)])
             (define a (if (even? i) 1 -1))
             (define b (if (even? (quotient i 2)) 1 -1))
             (list a (* (expt 2 10) (+ b (* (expt 2 -8) a)))))])
    (lcase "a column 2^-8 from orthogonal to one 2^10 wider"
           X
           (for/list ([row (in-list X)])
             (+ (* (expt 2 -21) (car row)) (* (expt 2 -30) (cadr row))))
           #f)))

(test-gpu "a float32 one-epoch stop 0.8% off is refused at 2^17 rows (Lo3)"
  (check-cuml-error 'logic
                    stop-message
                    (lambda () (en-case-fit nearly-orthogonal 'float32 #:alpha 0 #:tol 1e-6)))
  (match-define (list (list b0 _) _ 1)
    (en-case-fit nearly-orthogonal 'float32 #:alpha 0 #:tol 1e-6 #:max-iter 1))
  (check-true (> (abs (- (inexact->exact b0) (expt 2 -21))) (* 1e-3 (expt 2 -21)))))
