#lang racket/base

(require (only-in cuml/linear-model lasso-fit)
         (only-in math/flonum flnext flprev)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" unit-roundoff)
         (only-in "private/elastic-net-data.rkt"
                  apply-keywords
                  check-kkt-laws
                  default-tol
                  en-case-fit
                  en-run
                  en-run-coef
                  positive-rows-of)
         (only-in "private/elastic-net-oracle.rkt" alpha-max en-problem host-kkt kkt problem-of)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y)
         (only-in "private/real-linear-data.rkt" real-lcase))

(define glmnet-lambda-max 163.64265023355625)

(define rows
  '((1 alpha-max zeros) (1 glmnet zeros)
                        (1 ulp-above zeros)
                        (1 ulp-below zeros-or-fit)
                        (1 twice zeros)
                        (1 inside fit)
                        (1/2 alpha-max zeros)
                        (1/2 glmnet zeros)
                        (1/2 ulp-above zeros)
                        (1/2 ulp-below zeros-or-fit)
                        (1/2 twice zeros)
                        (1/2 inside fit)))

(define inside-share 0.95)

(define early-zeros '((1 9/10) (1/2 9/10)))

(define (point-alpha point top r)
  (match point
    ['alpha-max top]
    ['glmnet (/ glmnet-lambda-max (exact->inexact r))]
    ['ulp-above (flnext top)]
    ['ulp-below (flprev top)]
    ['twice (* 2 top)]
    ['inside (* inside-share top)]))

(define (fit-at c dt r alpha)
  (define options (hasheq '#:alpha alpha '#:tol default-tol))
  (apply-keywords en-case-fit
                  (if (= r 1)
                      (hash-set options '#:fit lasso-fit)
                      (hash-set options '#:l1-ratio r))
                  (list c dt)))

(define (check-zeros c dt p run)
  (match-define (struct* en-run ([coef coef] [intercept b])) run)
  (match-define (struct* en-problem ([my my])) p)
  (check-equal? coef (map (lambda (_) 0.0) coef))
  (check-true (<= (abs (- (inexact->exact b) my)) (* 2 (inexact->exact (unit-roundoff dt)) (abs my)))
              (format "intercept ~a, weighted mean ~a" b (exact->inexact my))))

(test-gpu "wine lasso and elastic net at and past alpha-max return zeros and the mean, never a refusal"
  (for* ([dt (in-list '(float32 float64))]
         [row (in-list rows)])
    (match-define (list r point expect) row)
    (define c (real-lcase 'wine '(raw . unweighted) dt))
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define alpha (point-alpha point (exact->inexact (alpha-max p r)) r))
    (test-case (format "~a l1-ratio ~a ~a alpha ~a" dt r point alpha)
      (define run (fit-at c dt r alpha))
      (define zeros? (andmap zero? (en-run-coef run)))
      (match expect
        ['zeros (check-zeros c dt p run)]
        ['zeros-or-fit
         (when zeros?
           (check-zeros c dt p run))]
        ['fit (check-true (not zeros?) "a nonzero fit")])
      (check-kkt-laws p run alpha r default-tol (positive-rows-of c) (unit-roundoff dt)))))

(test-case "zeros at 0.9 alpha-max fail the KKT law in both dtypes, so a zero fit accepted there is caught"
  (for* ([dt (in-list '(float32 float64))]
         [row (in-list early-zeros)])
    (match-define (list r share) row)
    (define c (real-lcase 'wine '(raw . unweighted) dt))
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define alpha (* (exact->inexact share) (exact->inexact (alpha-max p r))))
    (match-define (kkt K B A)
      (host-kkt p (map (lambda (_) 0.0) (car (lcase-X c))) alpha r default-tol 1 (positive-rows-of c) (unit-roundoff dt)))
    (check-true (> K (+ B A)) (format "~a l1-ratio ~a: residual ~a, bound ~a" dt r K (+ B A)))))
