#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in "private/elastic-net-oracle.rkt"
                  alpha-max
                  en-solution-coef
                  oracle-elastic-net
                  problem-of)
         (only-in "private/linear-data.rkt" docs-case lcase-X lcase-y))

(test-case "the oracle solves the docs data's lasso and zeroes it past alpha-max"
  (define p (problem-of (lcase-X docs-case) (lcase-y docs-case) #f #t))
  (check-equal? (alpha-max p 1) 5/4)
  (check-equal? (alpha-max p 1/2) 5/2)
  (match-define (list b0 _) (en-solution-coef (oracle-elastic-net p 0.1 1)))
  (check-true (< (abs (- b0 0.6)) 1e-12))
  (check-equal? (en-solution-coef (oracle-elastic-net p 3 1)) '(0 0)))

(test-case "the oracle's support solve is exact"
  (define p (problem-of (lcase-X docs-case) (lcase-y docs-case) #f #t))
  (check-equal? (en-solution-coef (oracle-elastic-net p 1/10 1)) '(3/5 2)))

(test-case "every coefficient the oracle returns is exact"
  (for* ([fit-intercept (in-list '(#t #f))]
         [l1-ratio (in-list '(1 1/2 1/20 0))]
         [alpha (in-list '(1/10 0.1 1/1000 1 3))])
    (define p (problem-of (lcase-X docs-case) (lcase-y docs-case) #f fit-intercept))
    (check-pred (lambda (coef) (andmap exact? coef))
                (en-solution-coef (oracle-elastic-net p alpha l1-ratio)))))
