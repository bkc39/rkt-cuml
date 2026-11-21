#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in cuml/linear-model/private/native libcuml-version)
         (only-in racket/hash hash-union)
         racket/runtime-path
         (only-in racket/string string-replace)
         (only-in "common-data.rkt" on-vector)
         (only-in "elastic-net-data.rkt" apply-keywords)
         (only-in "linear-data.rkt" on-matrix)
         (only-in "mbsgd-data.rkt" clf-parts reg-parts)
         (only-in "python-env.rkt" bits->floats floats->bits matrix->bits twin-results))

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
  (hash-union base more #:combine (lambda (_ v) v)))

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
  (overridden (hasheq 'op op
                      'dtype (symbol->string dt)
                      'n (length X)
                      'X (matrix->bits dt X)
                      'y (if y
                             (floats->bits dt y)
                             'null)
                      'codes (or codes 'null)
                      'loss (python-value loss)
                      'seed (or seed 'null))
              (for/hasheq ([(k v) (in-hash (settled settings))])
                (values (python-name k) (python-value v)))))

(define (mbsgd-twin-results requests)
  (twin-results mbsgd-twin-script requests #:expected-cuml libcuml-version))

(define (their-parts theirs dt)
  (list (bits->floats dt (hash-ref theirs 'coef_bits))
        (car (bits->floats dt (hash-ref theirs 'intercept_bits)))))
