#lang racket/base

(require (for-syntax racket/base)
         (only-in json json-null)
         (only-in racket/match match-define)
         (only-in racket/runtime-path define-runtime-path)
         (only-in "python-env.rkt" run-twin))

(provide case-of
         twin-results)

(define-runtime-path twin "../python/metrics_twin.py")

(define (twin-results cases)
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases cases)))
  (values results version))

(define (case-of op y p w #:dtype dt #:pred-dtype [pt dt])
  (hasheq 'op op
          'y_true y
          'y_pred p
          'dtype_true (symbol->string dt)
          'dtype_pred (symbol->string pt)
          'weight (or w (json-null))))
