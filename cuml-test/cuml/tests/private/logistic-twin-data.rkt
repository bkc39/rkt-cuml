#lang racket/base

(require (only-in racket/list append*)
         racket/runtime-path
         (only-in threading ~>)
         (only-in "linear-twin-data.rkt" bits-of decoded twin-results)
         (only-in "logistic-data.rkt"
                  default-tol
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  lr-fit
                  setting-C
                  setting-l1-ratio
                  setting-penalty))

(provide (struct-out twin-trial)
         logistic-twin
         logistic-twin-results
         setting-twin
         their-fit)

(struct twin-trial (dt case setting fit-intercept))

(define-runtime-path logistic-twin-script "../python/logistic_regression_twin.py")

(define (logistic-twin op
                       c
                       dt
                       #:penalty [penalty 'l2]
                       #:C [C 1.0]
                       #:l1-ratio [ratio #f]
                       #:fit-intercept [fi #t]
                       #:tol [tol default-tol]
                       #:max-iter [max-iter 1000])
  (hasheq 'op op
          'dtype (symbol->string dt)
          'X (bits-of dt (~> c lr-case-X append* (map exact->inexact _)))
          'y (lr-case-y c)
          'w (if (lr-case-w c)
                 (bits-of dt (map exact->inexact (lr-case-w c)))
                 'null)
          'penalty (symbol->string penalty)
          'C C
          'l1_ratio (or ratio 'null)
          'fit_intercept fi
          'tol tol
          'max_iter max-iter))

(define (setting-twin op
                      c
                      dt
                      s
                      #:fit-intercept [fi #t]
                      #:tol [tol default-tol]
                      #:max-iter [max-iter 1000])
  (logistic-twin op
                 c
                 dt
                 #:penalty (setting-penalty s)
                 #:C (setting-C s)
                 #:l1-ratio (setting-l1-ratio s)
                 #:fit-intercept fi
                 #:tol tol
                 #:max-iter max-iter))

(define (logistic-twin-results requests)
  (twin-results logistic-twin-script requests))

(define (their-fit theirs dt)
  (lr-fit (list (decoded dt (hash-ref theirs 'coef_bits)))
          (decoded dt (hash-ref theirs 'intercept_bits))
          (hash-ref theirs 'n_iter)))
