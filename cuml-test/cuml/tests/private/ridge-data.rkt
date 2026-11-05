#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-result-coef ridge-result-intercept)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "linear-data.rkt"
                  check-law
                  dtypes
                  exact-fit
                  gap
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  on-matrix
                  on-vector
                  ridge-oracles
                  ridge-solution
                  unit-roundoff
                  weights-or-ones))

(provide alpha-ratios
         check-ridge-laws
         lstsq-algorithm
         norm2
         ridge-case-fit
         ridge-case-result
         ridge-host-cosine
         ridge-result-fit
         solvers)

(define alpha-ratios (list (expt 2 -20) (expt 2 -7) 1 (expt 2 7)))
(define solvers '(eig svd))

(define (lstsq-algorithm solver)
  (if (eq? solver 'svd) 'svd-qr 'eig))

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(define (ridge-case-result c dt #:alpha alpha #:solver [solver 'auto] #:fit-intercept [fi #t])
  (ridge-fit (on-matrix (lcase-X c) dt)
             (on-vector (lcase-y c) dt)
             #:sample-weight (on-vector (lcase-w c) dt)
             #:alpha alpha
             #:solver solver
             #:fit-intercept fi))

(define (ridge-result-fit r)
  (list (~> r ridge-result-coef device-vector->list) (ridge-result-intercept r)))

(define (ridge-case-fit c dt #:alpha alpha #:solver [solver 'auto] #:fit-intercept [fi #t])
  (ridge-result-fit (ridge-case-result c dt #:alpha alpha #:solver solver #:fit-intercept fi)))

(define (ridge-host-cosine X y w coef intercept fi alpha)
  (define weights (weights-or-ones w y))
  (define total (apply + weights))
  (define (centre xs)
    (define m
      (if fi
          (/ (for/sum ([x (in-list xs)] [wi (in-list weights)]) (* wi x)) total)
          0))
    (map (lambda (x) (- x m)) xs))
  (define r
    (for/list ([row (in-list X)]
               [yi (in-list y)])
      (- yi intercept (for/sum ([x (in-list row)] [cj (in-list coef)]) (* x cj)))))
  (define y-norm (sqrt (for/sum ([v (in-list (centre y))] [wi (in-list weights)]) (* wi v v))))
  (for/fold ([worst 0.0])
            ([col (in-list (apply map list X))]
             [cj (in-list coef)])
    (define xc (centre col))
    (define num
      (- (for/sum ([x (in-list xc)] [ri (in-list r)] [wi (in-list weights)]) (* wi x ri))
         (* alpha cj)))
    (define square (+ alpha (for/sum ([x (in-list xc)] [wi (in-list weights)]) (* wi x x))))
    (if (zero? square)
        worst
        (max worst (exact->inexact (/ (abs num) (* (sqrt square) y-norm)))))))

(define (check-ridge-laws cases intercepts)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define y∞ (max-abs (lcase-y c)))
    (for* ([o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi alpha-ratios))]
           [dt (in-list dtypes)]
           [s (in-list solvers)])
      (match-define (ridge-solution alpha c* b* kappa _ _ _) o)
      (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
        (define u (unit-roundoff dt))
        (match-define (list coef b)
          (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
        (check-law "R1" (gap coef c*) (* 128 u kappa (max-abs c*)))
        (check-law "R3"
                   (ridge-host-cosine (lcase-X c) (lcase-y c) (lcase-w c) coef b fi alpha)
                   (* 64 u kappa))
        (when fi
          (check-law "R2" (abs (- b b*)) (* 8 u kappa y∞)))))))
