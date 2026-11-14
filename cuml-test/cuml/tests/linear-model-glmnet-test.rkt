#lang racket/base

(require (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" default-max-iter default-tol en-case-fit en-run)
         (only-in "private/elastic-net-oracle.rkt" alpha-max problem-of)
         (only-in "private/glmnet-data.rkt" glmnet-gaussian integer-weighted narrowed-lcase)
         (only-in "private/linear-data.rkt"
                  by-dtype
                  dtypes
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  normal-case)
         (only-in "private/logistic-data.rkt" relative-gap)
         (only-in "private/real-linear-data.rkt" real-lcase)
         (only-in "private/ridge-data.rkt" ridge-case-result ridge-result-fit))

(struct trial
        (case float32-bound
          float64-bound
          default-bound))

(define l1-ratios '(1 1/2 1/10))
(define alpha-shares '(1/3 1/30 1/300))
(define tight-max-iter 100000)

(define (tight-tol dt)
  (by-dtype dt 1e-7 1e-10))

(define (bound-of t dt)
  (by-dtype dt (trial-float32-bound t) (trial-float64-bound t)))

(define (total-weight c)
  (apply + (or (lcase-w c) (make-list (length (lcase-y c)) 1))))

(define (alpha-at c fi share)
  (exact->inexact (* share (alpha-max (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi) 1))))

(define (en-flat c dt alpha ratio fi tol max-iter)
  (match-define (en-run coef intercept _)
    (en-case-fit c
                 dt
                 #:alpha alpha
                 #:l1-ratio ratio
                 #:fit-intercept fi
                 #:tol tol
                 #:max-iter max-iter))
  (append coef (list intercept)))

(define (ridge-flat c dt alpha fi)
  (match-define (list coef intercept)
    (ridge-result-fit (ridge-case-result c dt #:alpha (* alpha (total-weight c)) #:fit-intercept fi)))
  (append coef (list intercept)))

(define (check-gaussian t dt)
  (define c (narrowed-lcase (trial-case t) dt))
  (for* ([fi (in-list '(#t #f))]
         [share (in-list alpha-shares)])
    (define alpha (alpha-at c fi share))
    (for ([ratio (in-list l1-ratios)])
      (test-case (format "~a ~a l1-ratio ~a alpha ~a intercept ~a" dt (lcase-name c) ratio alpha fi)
        (define theirs (glmnet-gaussian c alpha ratio #:fit-intercept fi))
        (define gap (relative-gap (en-flat c dt alpha ratio fi (tight-tol dt) tight-max-iter) theirs))
        (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap))
        (define default-gap
          (relative-gap (en-flat c dt alpha ratio fi default-tol default-max-iter) theirs))
        (check-true (<= default-gap (trial-default-bound t))
                    (format "default-tol gap ~a" default-gap))))
    (test-case (format "~a ~a ridge alpha ~a intercept ~a" dt (lcase-name c) alpha fi)
      (define gap
        (relative-gap (ridge-flat c dt alpha fi) (glmnet-gaussian c alpha 0 #:fit-intercept fi)))
      (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap)))))

(test-gpu "lasso, elastic-net and ridge fits are glmnet's gaussian fits with the ridge part over sd(y)"
  (define trials
    (list (trial (normal-case 200 5 1) 2e-6 1e-8 1e-4)
          (trial (integer-weighted (normal-case 200 5 2) 7) 2e-6 1e-8 1e-4)
          (trial (real-lcase 'diabetes '(standardised . unweighted) 'float64) 1e-4 1e-4 1e-1)
          (trial (real-lcase 'diabetes '(standardised . weighted) 'float64) 1e-4 1e-4 1e-1)))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-gaussian t dt)))
