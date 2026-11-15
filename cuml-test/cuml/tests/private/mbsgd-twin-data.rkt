#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in racket/list append*)
         racket/runtime-path
         (only-in racket/string string-replace)
         (only-in threading ~>)
         (only-in "elastic-net-data.rkt" apply-keywords)
         (only-in "linear-data.rkt" on-matrix on-vector)
         (only-in "linear-twin-data.rkt" bits-of decoded twin-results)
         (only-in "mbsgd-data.rkt" clf-parts reg-parts))

(provide clf-fit
         clf-result
         reg-fit
         their-parts
         twin-request
         mbsgd-twin-results
         overridden)

(define-runtime-path mbsgd-twin-script "../python/mbsgd_twin.py")

(define python-defaults
  (hash '#:penalty 'l2
        '#:alpha 1e-4
        '#:l1-ratio 0.15
        '#:fit-intercept #t
        '#:epochs 1000
        '#:tol 1e-3
        '#:shuffle #t
        '#:learning-rate 'constant
        '#:eta0 1e-3
        '#:power-t 0.5
        '#:batch-size 32
        '#:n-iter-no-change 5))

(define (overridden base more)
  (for/fold ([all base]) ([(k v) (in-hash more)])
    (hash-set all k v)))

(define (settled settings)
  (overridden python-defaults settings))

(define (reg-fit dt X y settings #:random-state [seed #f])
  (reg-parts (apply-keywords mbsgd-regressor-fit
                             (hash-set (settled settings) '#:random-state seed)
                             (list (on-matrix X dt) (on-vector y dt)))))

(define (clf-result dt X codes loss settings)
  (apply-keywords mbsgd-classifier-fit
                  (hash-set (settled settings) '#:loss loss)
                  (list (on-matrix X dt) (on-vector codes 'int32))))

(define (clf-fit dt X codes loss settings)
  (clf-parts (clf-result dt X codes loss settings)))

(define (python-name k)
  (string->symbol (string-replace (keyword->string k) "-" "_")))

(define (python-value v)
  (cond
    [(symbol? v) (string-replace (symbol->string v) "-" "_")]
    [(and (real? v) (not (exact-integer? v))) (exact->inexact v)]
    [else v]))

(define (twin-request op
                      dt
                      X
                      settings
                      #:y [y #f]
                      #:codes [codes #f]
                      #:loss [loss 'squared-loss]
                      #:seed [seed #f])
  (for/fold ([request (hasheq 'op op
                              'dtype (symbol->string dt)
                              'n (length X)
                              'X (bits-of dt (~> X append* (map exact->inexact _)))
                              'y (if y
                                     (bits-of dt (map exact->inexact y))
                                     'null)
                              'codes (or codes 'null)
                              'loss (python-value loss)
                              'seed (or seed 'null))])
            ([(k v) (in-hash (settled settings))])
    (hash-set request (python-name k) (python-value v))))

(define (mbsgd-twin-results requests)
  (twin-results mbsgd-twin-script requests))

(define (their-parts theirs dt)
  (list (decoded dt (hash-ref theirs 'coef_bits))
        (car (decoded dt (hash-ref theirs 'intercept_bits)))))
