#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fitted?
                  linear-regression
                  linear-regression-model
                  linear-regression?
                  predict
                  score)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-params
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in cuml/metrics r2-score)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" listed printed round-trips)
         (only-in "private/linear-data.rkt"
                  algorithms
                  dtypes
                  lcase-X
                  lcase-w
                  lcase-y
                  normal-case
                  on-matrix
                  on-vector)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define changed (hasheq 'algorithm 'qr 'fit-intercept #f))

(test-case "L3: the default parameters, and every keyword round-trips"
  (check-equal? (linear-regression-params) (hasheq 'algorithm 'auto 'fit-intercept #t))
  (check-equal? (estimator-params (linear-regression)) (linear-regression-params))
  (check-equal? (round-trips linear-regression changed) changed))

(test-case "L3: printing, equality and set-params"
  (check-equal? (printed (linear-regression)) (make-list 3 "#<linear-regression>"))
  (check-equal? (format "~v" (linear-regression #:algorithm 'qr)) "#<linear-regression #:algorithm 'qr>")
  (check-equal? (linear-regression #:algorithm 'svd) (linear-regression #:algorithm 'svd))
  (check-not-equal? (linear-regression #:fit-intercept #f) (linear-regression))
  (check-equal? (estimator-set-params (linear-regression) #:algorithm 'qr) (linear-regression #:algorithm 'qr))
  (check-false (linear-regression? 'linear-regression))
  (check-false (linear-regression-model (linear-regression))))

(test-case "linear regression is an estimator, a predictor and a scorer"
  (check-equal? (capabilities (linear-regression) linear-regression?) '(#t #t #f #f #t #t)))

(define data (normal-case 300 5 0 #:weighted? #t))

(test-gpu "L1, L2: fit, predict and score are the procedures, bitwise"
  (for* ([dt (in-list dtypes)] [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define X (on-matrix (lcase-X data) dt))
      (define y (on-vector (lcase-y data) dt))
      (define w (on-vector (lcase-w data) dt))
      (define e (linear-regression #:algorithm a))
      (define fitted-e (fit e X y))
      (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
      (check-equal? (format "~v" fitted-e)
                    (if (eq? a 'auto)
                        "#<linear-regression fitted>"
                        (format "#<linear-regression fitted #:algorithm '~a>" a)))
      (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
      (define direct (linear-regression-fit X y #:algorithm a))
      (define model (linear-regression-model fitted-e))
      (check-true (bits-equal? (listed (linear-regression-result-coef model))
                               (listed (linear-regression-result-coef direct))))
      (check-equal? (linear-regression-result-intercept model) (linear-regression-result-intercept direct))
      (define p (predict fitted-e X))
      (check-true (bits-equal? (listed p) (listed (linear-regression-predict direct X))))
      (check-equal? (score fitted-e X y) (r2-score y p))
      (check-equal? (score fitted-e X y #:sample-weight w) (r2-score y p #:sample-weight w)))))

(test-gpu "L4: fit needs y and takes a sample weight"
  (define X (on-matrix (lcase-X data) 'float64))
  (define y (on-vector (lcase-y data) 'float64))
  (define w (on-vector (lcase-w data) 'float64))
  (check-cuml-error 'logic "fit: linear-regression needs y" (lambda () (fit (linear-regression) X)))
  (check-true (bits-equal? (listed (linear-regression-result-coef
                                    (linear-regression-model (fit (linear-regression) X y #:sample-weight w))))
                           (listed (linear-regression-result-coef
                                    (linear-regression-fit X y #:sample-weight w))))))
