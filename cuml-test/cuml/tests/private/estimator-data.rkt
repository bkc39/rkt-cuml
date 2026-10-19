#lang racket/base

(require (only-in cuml estimator-params))

(provide round-trips)

(define (round-trips constructor changed)
  (for/hasheq ([name (in-hash-keys (estimator-params (constructor)))])
    (define keyword (string->keyword (symbol->string name)))
    (define built (keyword-apply constructor (list keyword) (list (hash-ref changed name)) '()))
    (values name (hash-ref (estimator-params built) name))))
