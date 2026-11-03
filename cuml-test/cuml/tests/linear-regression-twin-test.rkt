#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-params
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in "private/linear-data.rkt"
                  algorithms
                  dtypes
                  lcase
                  lcase-X
                  lcase-name
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
         (only-in "private/python-env.rkt" run-twin test-twin twin-params))

(define-runtime-path twin "python/linear_regression_twin.py")

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

(define (grid)
  (for*/list ([c (in-list p1-cases)] [fi (in-list '(#t #f))] [dt (in-list dtypes)] [a (in-list algorithms)])
    (list c fi dt a)))

(test-twin "P1, P2: bitwise against Python cuML on the staged data, and within 128 u kappa^2 on the raw data"
  (define rows (grid))
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

(define (already-staged)
  (define c (normal-case 300 5 9 #:weighted? #t))
  (define (scaled xs)
    (define m (max-abs xs))
    (map (lambda (x) (/ x m 2/3)) xs))
  (define columns (apply map list (lcase-X c)))
  (define X (apply map list (map (lambda (col) (map (lambda (x) (/ (round (* x 1024)) 1024)) (scaled col))) columns)))
  (lcase "already staged"
         X
         (map (lambda (v) (/ (round (* v 1024)) 1024)) (scaled (lcase-y c)))
         (map (lambda (v) (/ v 2)) (lcase-w c))))

(test-twin "E8: on already-staged data the fit is raw Python cuML's, bitwise"
  (define c (already-staged))
  (define rows (for*/list ([fi (in-list '(#t #f))] [dt (in-list dtypes)] [a (in-list algorithms)])
                 (list fi dt a)))
  (for ([g (in-list rows)]
        [r (in-list (results (for/list ([g (in-list rows)])
                               (match-define (list fi dt a) g)
                               (request "cuml" c dt a fi))))])
    (match-define (list fi dt a) g)
    (test-case (format "~a ~a ~a" fi dt a)
      (define-values (_r coef b) (ours c dt a fi))
      (check-equal? (list coef b) (theirs dt r)))))

(test-twin "P3: sklearn's float64 fit is within T1's bound"
  (define rows (for*/list ([c (in-list p1-cases)] [fi (in-list '(#t #f))] [a (in-list algorithms)])
                 (list c fi a)))
  (for ([g (in-list rows)]
        [r (in-list (results (for/list ([g (in-list rows)])
                               (match-define (list c fi a) g)
                               (request "sklearn" c 'float64 a fi))))])
    (match-define (list c fi a) g)
    (test-case (format "~a ~a ~a" (lcase-name c) fi a)
      (define-values (_r coef b) (ours c 'float64 a fi))
      (define kappa (hash-ref r 'kappa))
      (define sk (theirs 'float64 r))
      (check-law "P3"
                 (max-abs (map (lambda (x y) (- (inexact->exact x) (inexact->exact y))) coef (car sk)))
                 (* 128 (unit-roundoff 'float64) kappa kappa (max 1 (max-abs (car sk))))))))

(test-twin "P4: predictions agree with CuPy's product"
  (define rows (for*/list ([c (in-list p1-cases)] [dt (in-list dtypes)]) (list c dt)))
  (define fits
    (for/list ([g (in-list rows)])
      (match-define (list c dt) g)
      (define-values (r coef b) (ours c dt 'auto #t))
      (list r coef b)))
  (define requests
    (for/list ([g (in-list rows)] [f (in-list fits)])
      (match-define (list c dt) g)
      (hash-set* (request "predict" c dt 'auto #t)
                 'coef (bits-of dt (cadr f))
                 'intercept (caddr f))))
  (for ([g (in-list rows)] [f (in-list fits)] [r (in-list (results requests))])
    (match-define (list c dt) g)
    (match-define (list fitted coef b) f)
    (test-case (format "~a ~a" (lcase-name c) dt)
      (define d (length coef))
      (define u (unit-roundoff dt))
      (define ours-p (device-vector->list (linear-regression-predict fitted (on-matrix (lcase-X c) dt))))
      (for ([row (in-list (lcase-X c))]
            [p (in-list ours-p)]
            [q (in-list (decoded dt (hash-ref r 'pred_bits)))])
        (define size (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))
        (check-law "P4" (abs (- (inexact->exact p) (inexact->exact q))) (* 8 d u (inexact->exact size)))))))

(test-twin "P5: the defaults match Python cuML's"
  (match-define (list theirs) (results (list (hasheq 'op "params"))))
  (check-equal? (twin-params theirs '(copy-x))
                (hasheq 'algorithm 'eig 'fit-intercept #t))
  (check-equal? (linear-regression-params) (hasheq 'algorithm 'auto 'fit-intercept #t)))
