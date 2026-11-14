#lang racket/base

(require (only-in cuml/tests/private/elastic-net-data default-max-iter default-tol en-case-fit en-run)
         (only-in cuml/tests/private/elastic-net-oracle alpha-max problem-of)
         (only-in cuml/tests/private/linear-data
                  dtypes
                  law-cases
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  rounded-case
                  weights-or-ones)
         (only-in cuml/tests/private/logistic-data fit-flat lr-fit relative-gap)
         (only-in cuml/tests/private/real-linear-data real-lcase tight-max-iter tight-tol)
         (only-in cuml/tests/private/ridge-data ridge-case-result ridge-result-fit)
         (only-in racket/function const)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/glmnet-data.rkt" bound-of default-bound glmnet-gaussian trial trial-case whole-weighted))

(define l1-ratios '(1 1/2 1/10))
(define alpha-shares '(1/3 1/30 1/300))
(define replicated-rows-cap 1000)

(define (alpha-at c fi share)
  (~> (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi) (alpha-max 1) (* share) exact->inexact))

(define (en-fit c dt alpha ratio fi #:tol tol #:max-iter max-iter)
  (match-define (en-run coef intercept _)
    (en-case-fit c
                 dt
                 #:alpha alpha
                 #:l1-ratio ratio
                 #:fit-intercept fi
                 #:tol tol
                 #:max-iter max-iter))
  (lr-fit (list coef) (list intercept) #f))

(define (ridge-fit c dt alpha fi)
  (define total (apply + (weights-or-ones (lcase-w c) (lcase-y c))))
  (match-define (list coef intercept)
    (ridge-result-fit (ridge-case-result c dt #:alpha (* alpha total) #:fit-intercept fi)))
  (lr-fit (list coef) (list intercept) #f))

(define (gap-of ours theirs)
  (relative-gap (fit-flat ours) (fit-flat theirs)))

(define (check-elastic-net t dt c alpha ratio fi)
  (test-case (format "~a ~a l1-ratio ~a alpha ~a intercept ~a" dt (lcase-name c) ratio alpha fi)
    (define theirs (glmnet-gaussian c alpha ratio #:fit-intercept fi))
    (define gap (gap-of (en-fit c dt alpha ratio fi #:tol (tight-tol dt) #:max-iter tight-max-iter) theirs))
    (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap))
    (define cap (default-bound t dt))
    (when cap
      (define default-gap
        (gap-of (en-fit c dt alpha ratio fi #:tol default-tol #:max-iter default-max-iter) theirs))
      (check-true (<= default-gap cap) (format "default-tol gap ~a" default-gap)))))

(define (check-ridge t dt c alpha fi)
  (test-case (format "~a ~a ridge alpha ~a intercept ~a" dt (lcase-name c) alpha fi)
    (define gap (gap-of (ridge-fit c dt alpha fi) (glmnet-gaussian c alpha 0 #:fit-intercept fi)))
    (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap))))

(define (check-gaussian t dt)
  (define c (rounded-case (trial-case t) dt))
  (for* ([fi (in-list '(#t #f))]
         [share (in-list alpha-shares)])
    (define alpha (alpha-at c fi share))
    (for ([ratio (in-list l1-ratios)])
      (check-elastic-net t dt c alpha ratio fi))
    (check-ridge t dt c alpha fi)))

(define (law-trial c)
  (trial (if (lcase-w c)
             (whole-weighted c)
             c)
         #:float32 2e-6
         #:float64 (* 10 (tight-tol 'float64))
         #:default (const (* 10 default-tol))))

(define (replicable? c)
  (or (not (lcase-w c)) (<= (length (lcase-y c)) replicated-rows-cap)))

(test-gpu "lasso, elastic-net and ridge fits are glmnet's gaussian fits with the ridge part over sd(y)"
  (define trials
    (append (map law-trial (filter replicable? (law-cases)))
            (for/list ([weights (in-list '(unweighted weighted))])
              (trial (real-lcase 'diabetes (cons 'standardised weights) 'float64)
                     #:float32 1e-4
                     #:float64 5e-8
                     #:default (const 1e-1)))))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-gaussian t dt)))
