#lang racket/base

(require (only-in "private/real-linear-data.rkt" offset-law-arms test-real-set))

(for ([dt (in-list '(float32 float64))])
  (test-real-set 'breast-cancer
                 #:dtypes (list dt)
                 #:arms (hash-ref offset-law-arms dt)
                 #:checks '(lstsq)))
