#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-result-coef ridge-result-intercept)
         (only-in racket/match match-define struct*)
         (only-in rackunit test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "linear-data.rkt"
                  check-law-case
                  columns-of
                  dtypes
                  exact-fit
                  gap
                  host-cosine
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
                  weighted-spread
                  weights-or-ones))

(provide alpha-ratios
         check-ridge-fit
         check-ridge-laws
         lstsq-algorithm
         ridge-case-fit
         ridge-case-result
         ridge-result-fit
         rounding-alpha
         solvers
         widest-spread)

(define alpha-ratios (list (expt 2 -20) (expt 2 -7) 1 (expt 2 7)))
(define solvers '(eig svd))

(define (lstsq-algorithm solver)
  (if (eq? solver 'svd) 'svd 'eig))

(define (column-spreads c fi)
  (define weights (weights-or-ones (lcase-w c) (lcase-y c)))
  (define (spread col)
    (if fi
        (weighted-spread col weights)
        (for/sum ([x (in-list col)] [w (in-list weights)]) (* w x x))))
  (filter positive? (map spread (columns-of (lcase-X c)))))

(define (widest-spread c fi)
  (apply max (column-spreads c fi)))

(define (narrowest-spread c fi)
  (apply min (column-spreads c fi)))

(define (rounding-alpha c dt fi)
  (* (inexact->exact (unit-roundoff dt)) (narrowest-spread c fi)))

(define (ridge-case-result c dt #:alpha alpha #:solver [solver 'auto] #:fit-intercept [fi #t])
  (ridge-fit (on-matrix (lcase-X c) dt)
             (on-vector (lcase-y c) dt)
             #:sample-weight (on-vector (lcase-w c) dt)
             #:alpha alpha
             #:solver solver
             #:fit-intercept fi))

(define (ridge-result-fit r)
  (list (~> r ridge-result-coef device-vector->list) (ridge-result-intercept r)))

(define ridge-case-fit (compose1 ridge-result-fit ridge-case-result))

(define (eig-envelope? dt solver kappa₀)
  (and (eq? dt 'float32)
       (memq solver '(auto eig))
       (>= (* (inexact->exact (unit-roundoff dt)) kappa₀) 1)))

(define (check-ridge-fit c fi dt solver fit exact)
  (match-define (struct* ridge-solution
                         ([alpha alpha] [coef c*] [intercept b*] [kappa kappa] [kappa₀ kappa₀]))
    exact)
  (match-define (list coef b) (exact-fit fit))
  (define u (unit-roundoff dt))
  (define (enveloped factor bound)
    (if (eig-envelope? dt solver kappa₀)
        (* factor bound)
        bound))
  (check-law-case "R1" (gap coef c*) (enveloped 8 (* 128 u kappa (max-abs c*))))
  (check-law-case "R3"
                  (host-cosine (lcase-X c) (lcase-y c) (lcase-w c) coef b fi #:alpha alpha)
                  (enveloped 16 (* 64 u kappa)))
  (when fi
    (check-law-case "R2" (abs (- b b*)) (* 8 u kappa (max-abs (lcase-y c))))))

(define (check-ridge-laws cases intercepts)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (for* ([o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi alpha-ratios))]
           [dt (in-list dtypes)]
           [s (in-list solvers)])
      (match-define (struct* ridge-solution ([alpha alpha])) o)
      (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
        (check-ridge-fit c
                         fi
                         dt
                         s
                         (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)
                         o)))))

(module+ test
  (require (only-in rackunit check-false check-true))
  (check-true (eig-envelope? 'float32 'eig (expt 2 24)))
  (check-true (eig-envelope? 'float32 'auto +inf.0))
  (check-false (eig-envelope? 'float32 'eig (- (expt 2 24) 1)))
  (check-false (eig-envelope? 'float32 'svd +inf.0))
  (check-false (eig-envelope? 'float64 'eig +inf.0)))
