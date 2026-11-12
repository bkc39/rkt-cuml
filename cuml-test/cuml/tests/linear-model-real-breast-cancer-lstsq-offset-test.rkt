#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(define (offset-arms powers)
  (for*/list ([power (in-list powers)]
              [weighting (in-list '(unweighted weighted))])
    (cons (cons 'offset power) weighting)))

(test-real-set 'breast-cancer #:dtypes '(float32) #:arms (offset-arms '(10 12)) #:checks '(lstsq))
(test-real-set 'breast-cancer #:dtypes '(float64) #:arms (offset-arms '(10 12 14)) #:checks '(lstsq))
