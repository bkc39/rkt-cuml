#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  algorithms
                  dtypes
                  host-cosine
                  law-cases
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  on-matrix
                  on-vector
                  oracle-lstsq
                  solution
                  unit-roundoff
                  within))

(define cases (law-cases))

(define (fit-of X y w dt a fi)
  (define r
    (linear-regression-fit (on-matrix X dt)
                           (on-vector y dt)
                           #:sample-weight (on-vector w dt)
                           #:algorithm a
                           #:fit-intercept fi))
  (values r
          (map inexact->exact (device-vector->list (linear-regression-result-coef r)))
          (inexact->exact (linear-regression-result-intercept r))))

(define (gap xs ys)
  (max-abs (map - xs ys)))

(define (check-law name got bound)
  (check-true (<= (within got bound) 1) (format "~a: ~a of the bound" name (within got bound))))

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(test-gpu "T1-T4: against the exact-rational oracle"
  (for* ([c (in-list cases)] [fi (in-list '(#t #f))])
    (match-define (solution c* b* kappa) (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define y∞ (max-abs (lcase-y c)))
    (define weights (or (lcase-w c) (make-list (length (lcase-y c)) 1)))
    (for* ([dt (in-list dtypes)] [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (define u (unit-roundoff dt))
        (define-values (_r coef b) (fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt a fi))
        (check-law "T1" (gap coef c*) (* 128 u kappa kappa (max (max-abs c*) 1)))
        (check-law "T3"
                   (host-cosine (lcase-X c) (lcase-y c) (lcase-w c) coef b fi)
                   (* 64 u kappa))
        (when fi
          (check-law "T2" (abs (- b b*)) (* 8 u kappa y∞))
          (define residual-sum
            (for/sum ([row (in-list (lcase-X c))] [yi (in-list (lcase-y c))] [wi (in-list weights)])
              (* wi (- yi b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))))
          (check-law "T4"
                     (abs residual-sum)
                     (* 4 (sqrt (length weights)) u (norm2 (lcase-y c)) (max-abs weights))))))))

(test-gpu "T12: predictions are the exact host value to within the dot-product bound"
  (for* ([c (in-list cases)] [dt (in-list dtypes)] #:unless (lcase-w c))
    (test-case (format "~a ~a" (lcase-name c) dt)
      (define u (unit-roundoff dt))
      (define-values (r coef b) (fit-of (lcase-X c) (lcase-y c) #f dt 'auto #t))
      (define d (length coef))
      (define p (device-vector->list (linear-regression-predict r (on-matrix (lcase-X c) dt))))
      (for ([row (in-list (lcase-X c))] [pi (in-list p)])
        (define exact (+ b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))
        (define size (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))
        (check-law "T12" (abs (- (inexact->exact pi) exact)) (* 2 d u size))))))
