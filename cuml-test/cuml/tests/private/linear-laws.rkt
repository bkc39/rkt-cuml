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
                  check-law-case
                  columns-of
                  dtypes
                  exact-fit
                  gap
                  host-cosine
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  law-case
                  max-abs
                  on-matrix
                  oracle-lstsq
                  result-fit
                  solution
                  staging
                  staging-of
                  unit-roundoff
                  weights-or-ones))

(provide check-lstsq-fit
         check-oracle-laws
         check-prediction-law)

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(define (gamma k u)
  (/ (* k u) (- 1 (* k u))))

(define (mean-roundings w)
  (if w 32 17))

(define (check-lstsq-fit c fi u fit o s)
  (match-define (solution c* b* _) o)
  (match-define (staging scale y-scale kappa) s)
  (match-define (list coef b) (exact-fit fit))
  (define (staged cs)
    (map (lambda (cj dj) (/ (* y-scale cj) dj)) cs scale))
  (define t1 (* 128 u kappa kappa (max (max-abs (staged c*)) 1)))
  (check-law-case "T1" (gap (staged coef) (staged c*)) t1)
  (check-law-case "T3" (host-cosine (lcase-X c) (lcase-y c) (lcase-w c) coef b fi) (* 64 u kappa))
  (when fi
    (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
    (define total (apply + weights))
    (define columns (columns-of (lcase-X c)))
    (define t4
      (+ (* 4 (sqrt (length weights)) u (norm2 (lcase-y c)) (max-abs weights))
         (* u total (abs b))
         (* (+ (gamma (mean-roundings (lcase-w c)) u) (gamma (length coef) u))
            (for/sum ([column (in-list columns)] [cj (in-list coef)])
                     (* (dot weights (map abs column)) (abs cj))))))
    (define staged-means
      (for/sum ([column (in-list columns)] [dj (in-list scale)])
               (* (abs (/ (dot weights column) total)) dj)))
    (check-law-case "T2" (abs (- b b*)) (+ (/ t4 total) (/ (* staged-means t1) y-scale)))
    (law-case "T4"
      (define residual-sum
        (for/sum ([row (in-list (lcase-X c))] [yi (in-list (lcase-y c))] [wi (in-list weights)])
                 (* wi (- yi b (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj))))))
      (check-law "T4" (abs residual-sum) t4))))

(define (check-oracle-laws cases intercepts)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define o (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (define s (staging-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (for* ([dt (in-list dtypes)]
           [a (in-list algorithms)])
      (test-case (format "~a ~a ~a ~a" (lcase-name c) fi dt a)
        (check-lstsq-fit c
                         fi
                         (unit-roundoff dt)
                         (case-fit c dt #:algorithm a #:fit-intercept fi)
                         o
                         s)))))

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
