#lang racket/base

(require (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/list make-list)
         (only-in racket/match match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in "common-data.rkt" unit-roundoff)
         (only-in "linear-data.rkt"
                  algorithms
                  case-fit
                  check-law
                  fit-gap
                  fit-size
                  kappa-of
                  lcase
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  normal-case
                  oracle-lstsq
                  rounded-case
                  scaled-columns
                  solution)
         (only-in "python-env.rkt" bits->floats floats->bits matrix->bits twin-results))

(provide case-request
         check-cuml-failures
         check-cuml-parity
         one-column
         p1-cases
         request
         results
         theirs
         with-constant)

(define-runtime-path twin "../python/linear_regression_twin.py")

(define (case-request op c dt fi settings)
  (for/fold ([request (hasheq 'op op
                              'dtype (symbol->string dt)
                              'X (matrix->bits dt (lcase-X c))
                              'y (floats->bits dt (lcase-y c))
                              'w (if (lcase-w c)
                                     (floats->bits dt (lcase-w c))
                                     'null)
                              'fit_intercept fi)])
            ([(key value) (in-hash settings)])
    (hash-set request key value)))

(define (request op c dt a fi)
  (case-request op
                c
                dt
                fi
                (hasheq 'algorithm (if (eq? a 'auto)
                                       "eig"
                                       (symbol->string a)))))

(define (results requests)
  (twin-results twin requests #:expected-cuml libcuml-version))

(define (theirs dt result)
  (match-define (list intercept) (bits->floats dt (hash-ref result 'intercept_bits)))
  (list (bits->floats dt (hash-ref result 'coef_bits)) intercept))

(define one-column (normal-case 200 1 7))

(define with-constant
  (let ([c (normal-case 200 4 8 #:weighted? #t)])
    (lcase "constant column"
           (map (lambda (row) (cons 5/4 row)) (lcase-X c))
           (lcase-y c)
           (lcase-w c))))

(define p1-cases
  (list (normal-case 200 7 0)
        (normal-case 200 7 1 #:weighted? #t)
        (normal-case 20 7 2)
        (normal-case 5000 7 3)
        (normal-case 1000 7 4 #:weighted? #t)
        (normal-case 5000 7 5 #:weighted? #t)
        with-constant
        one-column))

(define mean-order-roundings 4)

(define solve-roundings 2)

(define (check-cuml-parity dt)
  (define rows
    (for*/list ([c (in-list p1-cases)]
                [fi (in-list '(#t #f))]
                [a (in-list algorithms)])
      (list c fi dt a)))
  (define staged
    (results (for/list ([g (in-list rows)])
               (match-define (list c fi dt a) g)
               (request "cuml-staged" c dt a fi))))
  (define raw
    (results (for/list ([g (in-list rows)])
               (match-define (list c fi dt a) g)
               (request "cuml" c dt a fi))))
  (for ([g (in-list rows)]
        [s (in-list staged)]
        [r (in-list raw)])
    (match-define (list c fi dt a) g)
    (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
      (define fitted (case-fit c dt #:algorithm a #:fit-intercept fi))
      (define kappa (kappa-of c fi))
      (match (staged-law c fi dt fitted kappa)
        [(list name bound) (check-law name (fit-gap fitted (theirs dt s)) bound)]
        [#f (check-equal? fitted (theirs dt s) "staged, bitwise")])
      (unless (eq? c with-constant)
        (check-law "raw"
                   (fit-gap fitted (theirs dt r))
                   (* 128 (unit-roundoff dt) kappa kappa (fit-size fitted)))))))

(define (staged-law c fi dt fitted kappa)
  (define u (unit-roundoff dt))
  (define size (fit-size fitted))
  (cond
    [(eq? c one-column) (list "one column, staged" (* 4 u kappa size))]
    [(and fi (eq? dt 'float64))
     (match-define (cons row _) (lcase-X c))
     (define d (length row))
     (list
      "float64 with an intercept, staged"
      (* u size d (+ (* mean-order-roundings kappa (+ 3 kappa)) (* solve-roundings kappa kappa))))]
    [else #f]))

(module+ test
  (define (stub-gap c fi dt scale)
    (match-define (solution c* b _) (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define fitted (list (map real->double-flonum c*) (real->double-flonum b)))
    (define stub
      (list (map (lambda (x) (real->double-flonum (* scale x))) c*) (real->double-flonum b)))
    (match-define (list _ bound) (staged-law c fi dt fitted (kappa-of c fi)))
    (list (fit-gap fitted stub) bound))
  (for ([row (in-list '((#t float64 13/10 #f) (#t float64 1 #t)))])
    (match-define (list fi dt scale passes?) row)
    (test-case (format "with-constant ~a ~a, coefficients times ~a" fi dt scale)
      (match-define (list got bound) (stub-gap with-constant fi dt scale))
      (check-true (rational? bound) (format "the bound is ~a" bound))
      (check-equal? (<= got bound) passes? (format "gap ~a against ~a" got bound)))))

(define (staged-relative coef c* factors y-factor)
  (define (staged xs)
    (for/list ([x (in-list xs)]
               [s (in-list factors)])
      (/ (* (inexact->exact x) s) y-factor)))
  (/ (max-abs (map - (staged coef) (staged c*))) (max 1 (max-abs (staged c*)))))

(define (worst-column-error coef c*)
  (for/fold ([worst 0])
            ([x (in-list coef)]
             [c (in-list c*)]
             #:unless (zero? c))
    (max worst (/ (abs (- (inexact->exact x) c)) (abs c)))))

(define (scaled-case base factors y-factor)
  (rounded-case (scaled-columns base factors y-factor) 'float32))

(define (cuml-failures)
  (define base (normal-case 200 7 0 #:weighted? #t))
  (define constant-base (normal-case 200 4 8 #:weighted? #t))
  (define (scaled name factors y-factor failing)
    (list name (scaled-case base factors y-factor) base factors y-factor failing))
  (list (scaled "a column at 1e-5" (list 1 #e1e-5 1 1 1 1 1) 1 '(auto))
        (scaled "y at 1e37" (make-list 7 1) #e1e37 algorithms)
        (list "a constant column" with-constant constant-base (make-list 5 1) 1 '(qr))))

(define (check-cuml-failures)
  (define rows
    (for*/list ([failure (in-list (cuml-failures))]
                #:do [(match-define (list name c base factors y-factor failing) failure)]
                [a (in-list failing)])
      (list name c base factors y-factor a)))
  (for ([g (in-list rows)]
        [r (in-list (results (for/list ([g (in-list rows)])
                               (match-define (list _ c _ _ _ a) g)
                               (request "cuml" c 'float32 a #t))))])
    (match-define (list name c base factors y-factor a) g)
    (test-case (format "~a ~a" name a)
      (match-define (solution c* _ _) (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
      (match-define (list coef _) (theirs 'float32 r))
      (check-true (or (not (andmap rational? coef)) (> (worst-column-error coef c*) 1/2))
                  "Python cuML gets it wrong")
      (match-define (list ours _) (case-fit c 'float32 #:algorithm a))
      (define kappa (kappa-of base #t))
      (check-law "ours"
                 (staged-relative ours c* factors y-factor)
                 (* 128 (unit-roundoff 'float32) kappa kappa)))))
