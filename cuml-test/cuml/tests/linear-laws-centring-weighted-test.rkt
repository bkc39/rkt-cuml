#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in "private/common-data.rkt" columns-of unit-roundoff)
         (only-in "private/exact-algebra.rkt" dot)
         (only-in "private/linear-centring-data.rkt"
                  failed-laws
                  fails?
                  mean-radius
                  moved-solve
                  strided-weighted-mean
                  worst-y-epsilon)
         (only-in "private/linear-data.rkt" lcase lcase-X lcase-w lcase-y oracle-lstsq solution-coef)
         (only-in "private/real-linear-data.rkt" real-lcase))

(define corner-signs '(1 -1 -1 0 1 1 0 -1 0 -1 1 1 1 -1 -1 -1 -1 -1 0 -1 -1 1 1 1 -1 1 -1 1 1))

(test-case "breast-cancer float32 offset by 2^10 weighted, means at a point of their error box: T1 holds by C1 and fails without it"
  (define c (real-lcase 'breast-cancer '((offset . 10) . weighted) 'float32))
  (define u (unit-roundoff 'float32))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define radius (mean-radius c u))
  (define epsilon
    (for/list ([s (in-list corner-signs)]
               [column (in-list (columns-of (lcase-X c)))])
      (* s (radius column))))
  (define fit (moved-solve c epsilon (worst-y-epsilon c u epsilon)))
  (check-equal? (failed-laws c 'float32 fit exact) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:coef-centring? #f)))

(define (atomic-case)
  (define big (+ 1 (expt 2 -23)))
  (define half-ulp (+ (expt 2 -24) (expt 2 -44)))
  (define rows
    (for/fold ([rows '()]
               [sign 1]
               #:result (reverse rows))
              ([j (in-range (expt 2 15))])
      (define b (quotient j 32))
      (cond
        [(and (zero? (modulo j 32)) (< b 64))
         (values (cons (list 0 (/ (if (zero? b) big half-ulp) 2) 2) rows) sign)]
        [else (values (cons (list sign 0 1) rows) (- sign))])))
  (lcase "atomic partials"
         (for/list ([row (in-list rows)])
           (match-define (list x _ _) row)
           (list x))
         (for/list ([row (in-list rows)])
           (match-define (list _ target _) row)
           target)
         (for/list ([row (in-list rows)])
           (match-define (list _ _ weight) row)
           weight)))

(test-case "float32 weighted, 2^15 rows in 64 strided blocks, ȳ's block partials added largest first: c* = 0 and b = ŷ̄ exceed T4 under the coalesced count 32 and meet T1 to T4 under the strided count"
  (define c (atomic-case))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (check-equal? (solution-coef exact) '(0))
  (define b (strided-weighted-mean (lcase-w c) (lcase-y c)))
  (define u (inexact->exact (unit-roundoff 'float32)))
  (define weights (lcase-w c))
  (define coalesced-term
    (+ (* u (apply + weights) (abs b))
       (* (/ (* 32 u) (- 1 (* 32 u))) (dot weights (map abs (lcase-y c))))))
  (define residual-sum (for/sum ([yi (in-list (lcase-y c))] [wi (in-list weights)]) (* wi (- yi b))))
  (check-true (> (abs residual-sum) coalesced-term)
              (format "residual sum ~a of the coalesced-count term"
                      (exact->inexact (/ (abs residual-sum) coalesced-term))))
  (check-equal? (failed-laws c 'float32 (list '(0) b) exact) '()))
