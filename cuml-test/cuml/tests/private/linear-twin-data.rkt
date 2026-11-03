#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in "linear-data.rkt"
                  algorithms
                  lcase
                  lcase-name
                  lcase-X
                  lcase-w
                  lcase-y
                  max-abs
                  normal-case
                  on-matrix
                  on-vector
                  oracle-lstsq
                  solution-kappa
                  unit-roundoff
                  within)
         (only-in "python-env.rkt" run-twin))

(provide bits-of
         check-cuml-parity
         check-law
         decoded
         gap
         kappa-of
         one-column
         ours
         p1-cases
         request
         results
         theirs
         with-constant)

(define-runtime-path twin "../python/linear_regression_twin.py")

(define (bits-of dt xs)
  (define size (if (eq? dt 'float32) 4 8))
  (for/list ([x (in-list xs)])
    (integer-bytes->integer (real->floating-point-bytes x size) #f)))

(define (decoded dt bits)
  (define size (if (eq? dt 'float32) 4 8))
  (for/list ([b (in-list bits)])
    (floating-point-bytes->real (integer->integer-bytes b size #f))))

(define (request op c dt a fi)
  (hasheq 'op op
          'dtype (symbol->string dt)
          'X (bits-of dt (map exact->inexact (apply append (lcase-X c))))
          'y (bits-of dt (map exact->inexact (lcase-y c)))
          'w (if (lcase-w c) (bits-of dt (map exact->inexact (lcase-w c))) 'null)
          'algorithm (if (eq? a 'auto) "eig" (symbol->string a))
          'fit_intercept fi))

(define (results requests)
  (match-define (hash-table ('results results) ('version (hash-table ('cuml version))))
    (run-twin twin (hasheq 'cases requests)))
  (check-equal? version libcuml-version)
  results)

(define (ours c dt a fi)
  (define r
    (linear-regression-fit (on-matrix (lcase-X c) dt)
                           (on-vector (lcase-y c) dt)
                           #:sample-weight (on-vector (lcase-w c) dt)
                           #:algorithm a
                           #:fit-intercept fi))
  (values r
          (device-vector->list (linear-regression-result-coef r))
          (linear-regression-result-intercept r)))

(define (theirs dt result)
  (list (decoded dt (hash-ref result 'coef_bits))
        (car (decoded dt (hash-ref result 'intercept_bits)))))

(define (gap a b)
  (max-abs (map (lambda (x y) (- (inexact->exact x) (inexact->exact y)))
                (cons (cadr a) (car a))
                (cons (cadr b) (car b)))))

(define (check-law name got bound)
  (check-true (<= (within got bound) 1) (format "~a: ~a of the bound" name (within got bound))))

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
        (normal-case 2000 7 4 #:weighted? #t)
        (normal-case 5000 7 5 #:weighted? #t)
        with-constant
        one-column))

(define (kappa-of c fi)
  (solution-kappa (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi)))

(define (check-cuml-parity dt)
  (define rows
    (for*/list ([c (in-list p1-cases)] [fi (in-list '(#t #f))] [a (in-list algorithms)])
      (list c fi dt a)))
  (define staged (results (for/list ([g (in-list rows)])
                            (match-define (list c fi dt a) g)
                            (request "cuml-staged" c dt a fi))))
  (define raw (results (for/list ([g (in-list rows)])
                         (match-define (list c fi dt a) g)
                         (request "cuml" c dt a fi))))
  (for ([g (in-list rows)] [s (in-list staged)] [r (in-list raw)])
    (match-define (list c fi dt a) g)
    (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
      (define-values (_r coef b) (ours c dt a fi))
      (define fitted (list coef b))
      (define kappa (kappa-of c fi))
      (define u (unit-roundoff dt))
      (define size (max 1 (abs b) (max-abs coef)))
      (cond
        [(eq? c one-column)
         (check-law "P1 d'=1" (gap fitted (theirs dt s)) (* 4 u kappa size))]
        [(and fi (lcase-w c) (> (length (lcase-y c)) 2000))
         (check-law "P1 F11" (gap fitted (theirs dt s)) (* 128 u kappa kappa size))]
        [else (check-equal? fitted (theirs dt s) "P1 bitwise")])
      (unless (eq? c with-constant)
        (check-law "P2" (gap fitted (theirs dt r)) (* 128 u kappa kappa size))))))

