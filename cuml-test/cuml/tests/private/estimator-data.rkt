#lang racket/base

(require (only-in cuml estimator-params)
         (only-in racket/list combinations)
         (only-in racket/match match)
         (only-in raft device-matrix->list* device-vector->list device-vector?))

(provide apply-options
         hash-remove*
         keyword-of
         keywords-of
         listed
         printed
         round-trips
         subsets
         twice)

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

(define (subsets keys n)
  (for*/list ([k (in-range (add1 n))]
              [subset (in-list (combinations keys k))])
    subset))

(define (twice size)
  (* 2 size))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(define (keywords-of proc)
  (call-with-values (lambda () (procedure-keywords proc)) list))
