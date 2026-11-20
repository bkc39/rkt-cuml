#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit)
         (only-in racket/function curry)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/logistic-data.rkt"
                  balanced-class-weights
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
         (only-in "private/logistic-sklearn-twin.rkt" optimum-gap-bound)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results setting-twin)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt"
                  labelled
                  labelled-case
                  labelled-labels
                  labelled-set))

(define (balanced-twin set)
  (match-define (struct* labelled ((case c))) set)
  (define y (lr-case-y c))
  (setting-twin "sklearn-fit"
                (lr-case "balanced" (lr-case-X c) y (map (curry hash-ref (balanced-class-weights y)) y))
                'float64
                default-setting
                #:tol (tight-tol 'float64)))

(define (balanced-fit set dt #:tol [tol default-tol] #:max-iter [max-iter 1000])
  (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) dt)
                           (labelled-labels set)
                           #:class-weight 'balanced
                           #:tol tol
                           #:max-iter max-iter))

(struct arm (name set optimum tol bound))

(test-twin "wine: balanced class weights over unequal cultivars fit at sklearn's balanced optimum"
  (define raw (labelled-set 'wine "cultivar" '()))
  (define standardised (labelled-set 'wine "cultivar" '() #:standardise? #t))
  (match-define (list raw-optimum standardised-optimum)
    (logistic-twin-results (list (balanced-twin raw) (balanced-twin standardised))))
  (define arms
    (list (arm "raw, the default tol" raw raw-optimum (lambda (_) default-tol) default-tol-cap)
          (arm "standardised, tight" standardised standardised-optimum tight-tol optimum-gap-bound)))
  (define (gap-to optimum fitted)
    (relative-gap (gauged-fit (fit-parts fitted)) (gauged-sklearn optimum)))
  (for* ([dt (in-list dtypes)]
         [a (in-list arms)])
    (match-define (struct* arm ([name name] [set set] [optimum optimum] [tol tol] [bound bound])) a)
    (test-case (format "~a, ~a" name dt)
      (define gap (gap-to optimum (balanced-fit set dt #:tol (tol dt) #:max-iter tight-max-iter)))
      (check-true (<= gap (bound dt)) (format "coef gap ~a" gap)))))
