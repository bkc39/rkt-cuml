#lang racket/base

(require (only-in cuml estimator-params))

(provide apply-options
         round-trips)

(define (apply-options proc options . args)
  (define names (sort (hash-keys options) symbol<?))
  (keyword-apply proc
                 (map (lambda (name) (string->keyword (symbol->string name))) names)
                 (map (lambda (name) (hash-ref options name)) names)
                 args))

(define (round-trips constructor changed)
  (for/hasheq ([name (in-hash-keys (estimator-params (constructor)))])
    (define built (apply-options constructor (hasheq name (hash-ref changed name))))
    (values name (hash-ref (estimator-params built) name))))
