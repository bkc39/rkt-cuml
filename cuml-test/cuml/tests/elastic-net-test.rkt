#lang racket/base

(require (only-in cuml elastic-net elastic-net-model fit lasso lasso-model predict r2-score score)
         (only-in cuml/linear-model
                  elastic-net-fit
                  elastic-net-predict
                  elastic-net-result?
                  lasso-fit
                  lasso-predict
                  lasso-result?
                  linear-regression-predict
                  linear-regression-result)
         (only-in racket/list make-list)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt"
                  correlated-case
                  en-case-fit
                  en-fit
                  en-run
                  en-run-n-iter
                  law-alpha
                  selections)
         (only-in "private/elastic-net-oracle.rkt" problem-of)
         (only-in "private/linear-data.rkt"
                  decimals
                  docs-case
                  dtypes
                  insert-column
                  lcase
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  on-matrix
                  on-vector
                  remove-at
                  unit-roundoff
                  weights-or-ones))

(define intercept-rounding 4)

(test-gpu "the docs data fits as recorded, in float64"
  (define (fit-of #:alpha alpha #:l1-ratio r #:max-iter [k 1000])
    (en-case-fit docs-case 'float64 #:alpha alpha #:l1-ratio r #:max-iter k))
  (match-define (en-run lc lb lk) (fit-of #:alpha 0.1 #:l1-ratio 1))
  (check-equal? (decimals 6 (cons lb lc)) '("3.599023" "0.601953" "1.999023"))
  (check-equal? lk 11)
  (match-define (en-run ec eb ek) (fit-of #:alpha 0.1 #:l1-ratio 0.5))
  (check-equal? (decimals 6 (cons eb ec)) '("3.645965" "0.829906" "1.804588"))
  (check-equal? ek 9)
  (match-define (struct* en-run ([coef zc] [n-iter zk])) (fit-of #:alpha 0 #:l1-ratio 1))
  (check-equal? (decimals 6 zc) '("1.001953" "1.999023"))
  (check-equal? zk 11)
  (match-define (struct* en-run ([coef rc] [n-iter rk])) (fit-of #:alpha 0.1 #:l1-ratio 0))
  (check-equal? (decimals 6 rc) '("0.932454" "1.694811"))
  (check-equal? rk 8)
  (check-equal? (fit-of #:alpha 3 #:l1-ratio 1) (en-run '(0.0 0.0) 8.5 1))
  (match-define (struct* en-run ([coef mc] [n-iter mk]))
    (fit-of #:alpha 0.1 #:l1-ratio 1 #:max-iter 2))
  (check-equal? (decimals 6 mc) '("1.600000" "1.500000"))
  (check-equal? mk 2))

(test-gpu "lasso-fit is elastic-net-fit at l1-ratio 1, bitwise"
  (define c (correlated-case 200 7 0.9 1 #:weighted? #t))
  (define alpha (law-alpha (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t) 0.01 1))
  (for* ([dt (in-list dtypes)]
         [s (in-list selections)]
         [fi (in-list '(#t #f))])
    (check-equal? (en-case-fit c dt #:fit lasso-fit #:alpha alpha #:fit-intercept fi #:selection s)
                  (en-case-fit c dt #:alpha alpha #:l1-ratio 1 #:fit-intercept fi #:selection s))))

(test-gpu "a constant or zero column gets 0.0 and leaves the rest bitwise"
  (define c (correlated-case 200 4 0.5 2))
  (for* ([dt (in-list dtypes)]
         [s (in-list selections)]
         [fi (in-list '(#t #f))]
         [at (in-list '(0 2 4))])
    (define value (if fi 1e3 0))
    (define wider (lcase "wider" (insert-column (lcase-X c) at (lambda (_) value)) (lcase-y c) #f))
    (match-define (en-run coef b k) (en-case-fit c dt #:alpha 0.01 #:fit-intercept fi #:selection s))
    (match-define (en-run coef+ b+ k+)
      (en-case-fit wider dt #:alpha 0.01 #:fit-intercept fi #:selection s))
    (test-case (format "~a ~a ~a at ~a" dt s fi at)
      (check-equal? (list-ref coef+ at) 0.0)
      (check-equal? (remove-at coef+ at) coef)
      (check-equal? (list b+ k+) (list b k)))))

(test-gpu "max-iter caps n-iter, and past convergence leaves the fit bitwise"
  (define c (correlated-case 200 7 0.9 3))
  (for ([dt (in-list dtypes)])
    (define base (en-case-fit c dt #:alpha 0.001))
    (define n0 (en-run-n-iter base))
    (check-true (> n0 3))
    (for ([k (in-list (list n0 (add1 n0) (* 4 n0)))])
      (check-equal? (en-case-fit c dt #:alpha 0.001 #:max-iter k) base))
    (for ([k (in-list (list 1 (quotient n0 2) (sub1 n0)))])
      (check-equal? (en-run-n-iter (en-case-fit c dt #:alpha 0.001 #:max-iter k)) k))))

(test-gpu "alpha past alpha-max gives zeros in one epoch and the weighted mean"
  (define c (correlated-case 200 7 0.5 4 #:weighted? #t))
  (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
  (define weighted-mean
    (/ (for/sum ([w (in-list weights)] [v (in-list (lcase-y c))]) (* w v)) (apply + weights)))
  (for* ([fi (in-list '(#t #f))]
         [dt (in-list dtypes)]
         [r (in-list '(1 0.5))]
         [factor (in-list (list (+ 1 (expt 2 -8)) 2 (expt 2 20)))])
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define mean (if fi weighted-mean 0))
    (match-define (en-run coef b k)
      (en-case-fit c dt #:alpha (* factor (law-alpha p 1 r)) #:l1-ratio r #:fit-intercept fi))
    (test-case (format "~a l1-ratio ~a factor ~a fit-intercept ~a" dt r factor fi)
      (check-equal? coef (make-list 7 0.0))
      (check-equal? k 1)
      (check-true
       (<= (abs (- (inexact->exact b) mean))
           (* intercept-rounding (inexact->exact (unit-roundoff dt)) (max-abs (lcase-y c))))))))

(test-gpu "alpha 0 runs coordinate descent, and tol 0 runs max-iter epochs"
  (check-true (> (en-run-n-iter (en-case-fit docs-case 'float64 #:alpha 0)) 1))
  (check-equal? (en-run-n-iter (en-case-fit docs-case 'float64 #:alpha 0.1 #:tol 0 #:max-iter 37))
                37))

(test-gpu "predict, the estimator's model and score are the fit's, bitwise"
  (define c (correlated-case 200 7 0.5 5 #:weighted? #t))
  (define X (on-matrix (lcase-X c) 'float32))
  (define y (on-vector (lcase-y c) 'float32))
  (define w (on-vector (lcase-w c) 'float32))
  (define fitted (elastic-net-fit X y #:sample-weight w #:alpha 0.01))
  (define lasso-fitted (lasso-fit X y #:sample-weight w #:alpha 0.01))
  (define (as-ols result)
    (match-define (struct* en-run ([coef coef] [intercept b])) (en-fit result))
    (linear-regression-result (on-vector coef 'float32) b))
  (check-equal? (device-vector->list (elastic-net-predict fitted X))
                (device-vector->list (linear-regression-predict (as-ols fitted) X)))
  (check-equal? (device-vector->list (lasso-predict lasso-fitted X))
                (device-vector->list (linear-regression-predict (as-ols lasso-fitted) X)))
  (define e (fit (elastic-net #:alpha 0.01) X y #:sample-weight w))
  (define l (fit (lasso #:alpha 0.01) X y #:sample-weight w))
  (check-true (elastic-net-result? (elastic-net-model e)))
  (check-true (lasso-result? (lasso-model l)))
  (check-equal? (en-fit (elastic-net-model e)) (en-fit fitted))
  (check-equal? (en-fit (lasso-model l)) (en-fit lasso-fitted))
  (check-equal? (score e X y #:sample-weight w) (r2-score y (predict e X) #:sample-weight w))
  (check-equal? (device-vector->list (predict l X))
                (device-vector->list (lasso-predict lasso-fitted X)))
  (check-false (elastic-net-model (elastic-net)))
  (check-false (lasso-model (lasso))))
