#lang racket/base

(require (only-in cuml estimator-params)
         (only-in racket/match match)
         (only-in raft device-matrix->list* device-vector->list device-vector?))

(provide apply-options
         hash-remove*
         listed
         printed
         round-trips)

(define (apply-options proc options . args)
  (define names (sort (hash-keys options) symbol<?))
  (keyword-apply proc
                 (map (lambda (name) (string->keyword (symbol->string name))) names)
                 (map (lambda (name) (hash-ref options name)) names)
                 args))

(define (hash-remove* h keys)
  (for/fold ([h h]) ([k (in-list keys)])
    (hash-remove h k)))

(define (round-trips constructor changed)
  (for/hasheq ([name (in-hash-keys (estimator-params (constructor)))])
    (define built (apply-options constructor (hasheq name (hash-ref changed name))))
    (values name (hash-ref (estimator-params built) name))))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(define (listed v)
  (match v
    [#f #f]
    [(? device-vector?) (device-vector->list v)]
    [_ (device-matrix->list* v)]))
