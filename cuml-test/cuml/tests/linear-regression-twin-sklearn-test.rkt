#lang racket/base

(require (only-in cuml/linear-model linear-regression-params linear-regression-predict)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in "private/linear-data.rkt"
                  algorithms
                  case-fit
                  case-result
                  check-law
                  dtypes
                  gap
                  lcase
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  normal-case
                  on-matrix
                  result-fit
                  unit-roundoff)
         (only-in "private/linear-twin-data.rkt"
                  bits-of
                  decoded
                  one-column
                  p1-cases
                  request
                  results
                  theirs
                  with-constant)
         (only-in "private/python-env.rkt" test-twin twin-params))

(define (already-staged)
  (define c (normal-case 300 5 9 #:weighted? #t))
  (define (scaled xs)
    (define m (max-abs xs))
    (map (lambda (x) (/ x m 2/3)) xs))
  (define columns (apply map list (lcase-X c)))
  (define X
    (apply map
           list
           (map (lambda (col) (map (lambda (x) (/ (round (* x 1024)) 1024)) (scaled col))) columns)))
  (lcase "already staged"
         X
         (map (lambda (v) (/ (round (* v 1024)) 1024)) (scaled (lcase-y c)))
         (map (lambda (v) (/ v 2)) (lcase-w c))))

(test-twin "on already-staged data the fit is raw Python cuML's, bitwise"
  (define c (already-staged))
  (define rows
    (for*/list ([fi (in-list '(#t #f))]
                [dt (in-list dtypes)]
                [a (in-list algorithms)])
      (list fi dt a)))
  (for ([g (in-list rows)]
        [r (in-list (results (for/list ([g (in-list rows)])
                               (match-define (list fi dt a) g)
                               (request "cuml" c dt a fi))))])
    (match-define (list fi dt a) g)
    (test-case (format "~a ~a ~a" fi dt a)
      (check-equal? (case-fit c dt #:algorithm a #:fit-intercept fi) (theirs dt r)))))

(test-twin "sklearn's float64 fit is within the oracle bound"
  (define rows
    (for*/list ([c (in-list p1-cases)]
                [fi (in-list '(#t #f))]
                [a (in-list algorithms)])
      (list c fi a)))
  (for ([g (in-list rows)]
        [r (in-list (results (for/list ([g (in-list rows)])
                               (match-define (list c fi a) g)
                               (request "sklearn" c 'float64 a fi))))])
    (match-define (list c fi a) g)
    (test-case (format "~a ~a ~a" (lcase-name c) fi a)
      (match-define (list coef _) (case-fit c 'float64 #:algorithm a #:fit-intercept fi))
      (define kappa (hash-ref r 'kappa))
      (match-define (list sk _) (theirs 'float64 r))
      (check-law "sklearn"
                 (gap (map inexact->exact coef) (map inexact->exact sk))
                 (* 128 (unit-roundoff 'float64) kappa kappa (max 1 (max-abs sk)))))))

(test-twin "predictions agree with CuPy's product"
  (define rows
    (for*/list ([c (in-list p1-cases)]
                [dt (in-list dtypes)])
      (list c dt)))
  (define fits
    (for/list ([g (in-list rows)])
      (match-define (list c dt) g)
      (define r (case-result c dt))
      (cons r (result-fit r))))
  (define requests
    (for/list ([g (in-list rows)]
               [f (in-list fits)])
      (match-define (list c dt) g)
      (match-define (list _ coef b) f)
      (hash-set* (request "predict" c dt 'auto #t) 'coef (bits-of dt coef) 'intercept b)))
  (for ([g (in-list rows)]
        [f (in-list fits)]
        [r (in-list (results requests))])
    (match-define (list c dt) g)
    (match-define (list fitted coef b) f)
    (test-case (format "~a ~a" (lcase-name c) dt)
      (define d (length coef))
      (define u (unit-roundoff dt))
      (define ours-p
        (device-vector->list (linear-regression-predict fitted (on-matrix (lcase-X c) dt))))
      (for ([row (in-list (lcase-X c))]
            [p (in-list ours-p)]
            [q (in-list (decoded dt (hash-ref r 'pred_bits)))])
        (define size (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))
        (check-law "prediction"
                   (abs (- (inexact->exact p) (inexact->exact q)))
                   (* 8 d u (inexact->exact size)))))))

(test-twin "the defaults match Python cuML's"
  (match-define (list theirs) (results (list (hasheq 'op "params"))))
  (check-equal? (twin-params theirs '(copy-X)) (hasheq 'algorithm 'auto 'fit-intercept #t))
  (check-equal? (linear-regression-params) (hasheq 'algorithm 'auto 'fit-intercept #t)))
