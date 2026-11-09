#lang racket/base

(require (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in threading ~>)
         (only-in "linear-data.rkt"
                  algorithms
                  case-fit
                  check-law
                  fit-gap
                  fit-size
                  kappa-of
                  lcase
                  lcase-name
                  lcase-X
                  lcase-w
                  lcase-y
                  max-abs
                  normal-case
                  oracle-lstsq
                  rounded-case
                  scaled-columns
                  solution
                  unit-roundoff)
         (only-in "python-env.rkt" run-twin))

(provide bits-of
         case-request
         check-cuml-failures
         check-cuml-parity
         decoded
         one-column
         p1-cases
         request
         results
         theirs
         twin-results
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

(define (case-request op c dt fi settings)
  (for/fold ([request (hasheq 'op op
                              'dtype (symbol->string dt)
                              'X (bits-of dt (~> c lcase-X (apply append _) (map exact->inexact _)))
                              'y (bits-of dt (map exact->inexact (lcase-y c)))
                              'w (if (lcase-w c)
                                     (bits-of dt (map exact->inexact (lcase-w c)))
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

(define (twin-results script requests)
  (match-define (hash-table ('results results) ('version (hash-table ('cuml version))))
    (run-twin script (hasheq 'cases requests)))
  (check-equal? version libcuml-version)
  results)

(define (results requests)
  (twin-results twin requests))

(define (theirs dt result)
  (match-define (list intercept) (decoded dt (hash-ref result 'intercept_bits)))
  (list (decoded dt (hash-ref result 'coef_bits)) intercept))

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
      (define u (unit-roundoff dt))
      (define size (fit-size fitted))
      (cond
        [(eq? c one-column)
         (check-law "one column, staged" (fit-gap fitted (theirs dt s)) (* 4 u kappa size))]
        [(and fi (lcase-w c) (> (length (lcase-y c)) 2000))
         (check-law "weighted with an intercept, staged"
                    (fit-gap fitted (theirs dt s))
                    (* 128 u kappa kappa size))]
        [else (check-equal? fitted (theirs dt s) "staged, bitwise")])
      (unless (eq? c with-constant)
        (check-law "raw" (fit-gap fitted (theirs dt r)) (* 128 u kappa kappa size))))))

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
