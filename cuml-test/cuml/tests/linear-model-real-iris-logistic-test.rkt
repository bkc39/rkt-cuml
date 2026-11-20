#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit)
         (only-in racket/list count remove-duplicates)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in "private/linear-data.rkt" by-dtype dtypes on-matrix)
         (only-in "private/logistic-data.rkt"
                  default-setting
                  default-tol
                  default-tol-cap
                  fit-parts
                  gauged-fit
                  gauged-sklearn
                  lr-case
                  lr-case-X
                  lr-case-y
                  relative-gap
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results setting-twin)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt"
                  check-score
                  check-sklearn-optimum
                  labelled
                  labelled-case
                  labelled-labels
                  labelled-set))

(define iris (labelled-set 'iris "species" '()))

(test-twin "iris: the fit is at sklearn's optimum"
  (check-sklearn-optimum iris 'float64 (list default-setting)))

(test-twin "iris: the estimator scores as sklearn"
  (check-score iris))

(define (balanced-weights y)
  (define n (length y))
  (define k (length (remove-duplicates y)))
  (map (lambda (c) (/ n (* k (count (lambda (v) (= v c)) y)))) y))

(define (balanced-gap-bound dt)
  (by-dtype dt 2e-3 1e-6))

(define (balanced-twin set)
  (match-define (struct* labelled ((case c))) set)
  (define y (lr-case-y c))
  (setting-twin "sklearn-fit"
                (lr-case "balanced" (lr-case-X c) y (balanced-weights y))
                'float64
                default-setting
                #:tol (tight-tol 'float64)))

(define (balanced-fit set dt #:tol [tol default-tol] #:max-iter [max-iter 1000])
  (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) dt)
                           (labelled-labels set)
                           #:class-weight 'balanced
                           #:tol tol
                           #:max-iter max-iter))

(test-twin "wine: balanced class weights over unequal cultivars fit at sklearn's balanced optimum"
  (define raw (labelled-set 'wine "cultivar" '()))
  (define standardised (labelled-set 'wine "cultivar" '() #:standardise? #t))
  (match-define (list raw-optimum optimum)
    (logistic-twin-results (list (balanced-twin raw) (balanced-twin standardised))))
  (for ([dt (in-list dtypes)])
    (test-case (format "raw, the default tol, ~a" dt)
      (define gap
        (relative-gap (gauged-fit (fit-parts (balanced-fit raw dt #:max-iter tight-max-iter)))
                      (gauged-sklearn raw-optimum)))
      (check-true (<= gap (default-tol-cap dt)) (format "coef gap ~a" gap)))
    (test-case (format "standardised, tight, ~a" dt)
      (define fitted (balanced-fit standardised dt #:tol (tight-tol dt) #:max-iter tight-max-iter))
      (define gap (relative-gap (gauged-fit (fit-parts fitted)) (gauged-sklearn optimum)))
      (check-true (<= gap (balanced-gap-bound dt)) (format "coef gap ~a" gap)))))
