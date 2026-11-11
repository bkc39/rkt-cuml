#lang racket/base

(require (only-in cuml/linear-model linear-regression-predict)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "exact-algebra.rkt" dot)
         (only-in "linear-data.rkt"
                  algorithms
                  case-fit
                  case-result
                  check-law
                  columns-of
                  dtypes
                  exact-fit
                  gap
                  host-cosine
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  on-matrix
                  oracle-lstsq
                  result-fit
                  solution
                  unit-roundoff
                  weights-or-ones))

(provide check-lstsq-fit
         check-oracle-laws
         check-prediction-law)

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(define (check-lstsq-fit c fi u fit o)
  (match-define (solution c* b* kappa) o)
  (match-define (list coef b) (exact-fit fit))
  (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
  (check-law "T1" (gap coef c*) (* 128 u kappa kappa (max (max-abs c*) 1)))
  (check-law "T3" (host-cosine (lcase-X c) (lcase-y c) (lcase-w c) coef b fi) (* 64 u kappa))
  (when fi
    (check-law "T2" (abs (- b b*)) (* 8 u kappa (max-abs (lcase-y c))))
    (define residual-sum
      (for/sum ([row (in-list (lcase-X c))] [yi (in-list (lcase-y c))] [wi (in-list weights)])
               (* wi (- yi b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))))
    (define total (apply + weights))
    (define cancelled
      (for/sum ([column (in-list (columns-of (lcase-X c)))] [cj (in-list coef)])
               (abs (* (/ (dot weights column) total) cj))))
    (check-law "T4"
               (abs residual-sum)
               (+ (* 4 (sqrt (length weights)) u (norm2 (lcase-y c)) (max-abs weights))
                  (* u total (+ (abs b) cancelled))))))

(define (check-oracle-laws cases intercepts)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define o (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (for* ([dt (in-list dtypes)]
           [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (check-lstsq-fit c
                         fi
                         (unit-roundoff dt)
                         (case-fit c dt #:algorithm a #:fit-intercept fi)
                         o)))))

(define (check-prediction-law cases)
  (for* ([c (in-list cases)]
         [dt (in-list dtypes)]
         #:unless (lcase-w c))
    (test-case (format "~a ~a" (lcase-name c) dt)
      (define u (unit-roundoff dt))
      (define r (case-result c dt))
      (match-define (list coef b) (exact-fit (result-fit r)))
      (define d (length coef))
      (define p (~> r (linear-regression-predict (on-matrix (lcase-X c) dt)) device-vector->list))
      (for ([row (in-list (lcase-X c))]
            [pi (in-list p)])
        (define exact (+ b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))
        (define size (+ (abs b) (for/sum ([x (in-list row)] [cj (in-list coef)]) (abs (* x cj)))))
        (check-law "T12" (abs (- (inexact->exact pi) exact)) (* 2 d u size))))))
