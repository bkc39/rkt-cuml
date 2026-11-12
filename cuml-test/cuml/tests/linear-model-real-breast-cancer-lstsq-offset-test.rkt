#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'breast-cancer
               #:arms (for*/list ([power (in-list '(10 12 14))]
                                  [weighting (in-list '(unweighted weighted))])
                        (cons (cons 'offset power) weighting))
               #:checks '(lstsq))
