#lang racket/base

(require (only-in cuml/linear-model lasso-fit)
         (only-in math/flonum flnext flprev)
         (only-in racket/flonum flsingle)
         (only-in racket/list make-list)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
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
         (only-in "private/linear-data.rkt" lcase lcase-X lcase-w lcase-y)
         (only-in "private/real-linear-data.rkt" real-lcase))

(define glmnet-lambda-max 163.64265023355625)

(define inside-share 0.95)

(define (expected point)
  (match point
    ['ulp-below 'zeros-or-fit]
    ['inside 'fit]
    [_ 'zeros]))

(define rows
  (for*/list ([setup (in-list '((unweighted #t) (weighted #t) (unweighted #f)))]
              [r (in-list '(1 1/2))]
              [point (in-list '(alpha-max glmnet ulp-above ulp-below twice inside))]
              #:unless (and (eq? point 'glmnet) (not (equal? setup '(unweighted #t)))))
    (match-define (list weighting fi) setup)
    (list weighting fi r point (expected point))))

(define early-zeros '((1 9/10) (1/2 9/10)))

(define (wine dt weighting fi r)
  (define c (real-lcase 'wine (cons 'raw weighting) dt))
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
  (values c p (exact->inexact (alpha-max p r))))

(define (point-alpha point top r)
  (match point
    ['alpha-max top]
    ['glmnet (/ glmnet-lambda-max (exact->inexact r))]
    ['ulp-above (flnext top)]
    ['ulp-below (flprev top)]
    ['twice (* 2 top)]
    ['inside (* inside-share top)]))

(define (zeros-like xs)
  (make-list (length xs) 0.0))

(define (fit-at c dt r fi alpha)
  (define-values (key value)
    (if (= r 1)
        (values '#:fit lasso-fit)
        (values '#:l1-ratio r)))
  (apply-keywords en-case-fit
                  (hasheq '#:alpha alpha '#:fit-intercept fi '#:tol default-tol key value)
                  (list c dt)))

(define (single-step x k)
  (floating-point-bytes->real
   (integer->integer-bytes (+ k (integer-bytes->integer (real->floating-point-bytes x 4) #t)) 4 #t)))

(define (single-even? x)
  (even? (integer-bytes->integer (real->floating-point-bytes x 4) #f)))

(define (nearest-single q s)
  (define (distance v)
    (abs (- (inexact->exact v) q)))
  (for/fold ([best s])
            ([v (in-list (list (single-step s -1) (single-step s 1)))]
             #:when (rational? v))
    (define d (distance v))
    (define e (distance best))
    (if (or (< d e) (and (= d e) (single-even? v)))
        v
        best)))

(define (correctly-rounded dt q)
  (define near (real->double-flonum q))
  (match dt
    ['float64 near]
    ['float32
     (define s (flsingle near))
     (if (and (rational? s) (not (zero? s)))
         (nearest-single q s)
         s)]))

(for ([row (in-list `((float64 0 0.0)
                      (float64 -13/4 -3.25)
                      (float32 0 0.0)
                      (float32 -13 -13.0)
                      (float32 ,(expt 2 -140) ,(expt 2.0 -140))
                      (float32 ,(expt 2 -151) 0.0)
                      (float32 ,(- (expt 2 -151)) -0.0)
                      (float32 ,(expt 2 -160) 0.0)
                      (float32 ,(+ 1 (* 3 (expt 2 -24))) ,(+ 1.0 (expt 2.0 -22)))
                      (float32 ,(- -1 (* 3 (expt 2 -24))) ,(- -1.0 (expt 2.0 -22)))
                      (float32 ,(+ 1 (expt 2 -24) (expt 2 -60)) ,(+ 1.0 (expt 2.0 -23)))
                      (float32 1/3 ,(flsingle (/ 1.0 3)))))])
  (match-define (list dt q expected) row)
  (test-case (format "correctly-rounded ~a ~a is ~a, ties to even" dt (exact->inexact q) expected)
    (check-equal? (correctly-rounded dt q) expected)))

(define (check-zeros dt p run)
  (match-define (struct* en-run ([coef coef] [intercept b])) run)
  (match-define (struct* en-problem ([my my])) p)
  (check-equal? coef (zeros-like coef))
  (check-equal? b (correctly-rounded dt my)))

(test-gpu "wine lasso and elastic net at and past alpha-max return zeros and the mean, never a refusal"
  (for* ([dt (in-list '(float32 float64))]
         [row (in-list rows)])
    (match-define (list weighting fi r point expect) row)
    (define-values (c p top) (wine dt weighting fi r))
    (define alpha (point-alpha point top r))
    (test-case (format "~a ~a fit-intercept ~a l1-ratio ~a ~a alpha ~a" dt weighting fi r point alpha)
      (define run (fit-at c dt r fi alpha))
      (define zeros? (andmap zero? (en-run-coef run)))
      (match expect
        ['zeros (check-zeros dt p run)]
        ['zeros-or-fit
         (when zeros?
           (check-zeros dt p run))]
        ['fit (check-true (not zeros?) "a nonzero fit")])
      (check-kkt-laws p run alpha r default-tol (positive-rows-of c) (unit-roundoff dt)))))

(test-case "zeros at 0.9 alpha-max fail the KKT law in both dtypes, so a zero fit accepted there is caught"
  (for* ([dt (in-list '(float32 float64))]
         [row (in-list early-zeros)])
    (match-define (list r share) row)
    (define-values (c p top) (wine dt 'unweighted #t r))
    (define alpha (* (exact->inexact share) top))
    (match-define (kkt K B A)
      (host-kkt p
                (zeros-like (car (lcase-X c)))
                alpha
                r
                default-tol
                1
                (positive-rows-of c)
                (unit-roundoff dt)))
    (check-true (> K (+ B A)) (format "~a l1-ratio ~a: residual ~a, bound ~a" dt r K (+ B A)))))

(test-gpu "float32 just past the zero test's reach below alpha-max is refused as within the solve's rounding of it"
  (for ([r (in-list '(1 1/2))])
    (define-values (c _p top) (wine 'float32 'unweighted #t r))
    (define alpha (* top (- 1 (expt 2.0 -20))))
    (test-case (format "l1-ratio ~a alpha ~a" r alpha)
      (check-exn
       #rx"under float32's rounding of the solve, which no tol resolves; alpha [0-9.]+ is that close below alpha-max [0-9.]+, from which every coefficient is 0: raise alpha to alpha-max or lower it, or fit in float64$"
       (lambda () (fit-at c 'float32 r #t alpha))))))

(define narrow-spread (expt 2 -20))

(define (narrow-binding-case)
  (define a (inexact->exact (+ 0.9 (sqrt 0.19))))
  (define b (inexact->exact (- 0.9 (sqrt 0.19))))
  (define pattern `((1 1 ,a) (1 -1 ,b) (-1 1 ,(- a)) (-1 -1 ,(- b))))
  (define table
    (for*/list ([_ (in-range 250)]
                [row (in-list pattern)])
      row))
  (lcase "a narrow column with correlation 0.9 beside a wide orthogonal one"
         (for/list ([row (in-list table)])
           (match-define (list u v _) row)
           (list (* narrow-spread u) v))
         (for/list ([row (in-list table)])
           (match-define (list _ _ y) row)
           y)
         #f))

(test-gpu "a narrow column just inside alpha-max is fitted, not zeroed by a wide column's allowance"
  (define c (narrow-binding-case))
  (define p (problem-of (lcase-X c) (lcase-y c) #f #t))
  (define top (alpha-max p 1))
  (define alpha (* (exact->inexact top) (- 1 1.48e-7)))
  (define run (fit-at c 'float64 1 #t alpha))
  (match-define (list narrow wide) (en-run-coef run))
  (define fitted (/ (- top (inexact->exact alpha)) (* narrow-spread narrow-spread)))
  (check-true (< (abs (- (inexact->exact narrow) fitted)) (* 1/100 fitted))
              (format "narrow coefficient ~a, expected ~a" narrow (exact->inexact fitted)))
  (check-equal? wide 0.0)
  (check-kkt-laws p run alpha 1 default-tol (positive-rows-of c) (unit-roundoff 'float64)))
