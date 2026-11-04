#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/metrics/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/match match-define)
         (only-in racket/runtime-path define-runtime-path)
         (only-in rackunit check-equal?)
         (only-in "python-env.rkt" run-twin))

(provide case-of
         checked-twin-results)

(define-runtime-path twin "../python/metrics_twin.py")

(define (checked-twin-results cases)
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases cases)))
  (check-equal? version libcuml-version "T5")
  results)

(define (case-of op y p w #:dtype dt #:pred-dtype [pt dt])
  (hasheq 'op op
          'y_true y
          'y_pred p
          'dtype_true (symbol->string dt)
          'dtype_pred (symbol->string pt)
          'weight (or w (json-null))))
