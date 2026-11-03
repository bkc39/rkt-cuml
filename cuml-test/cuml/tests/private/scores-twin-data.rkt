#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/metrics/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/match match-define)
         (only-in racket/runtime-path define-runtime-path)
         (only-in rackunit check-equal?)
         (only-in raft list*->device-matrix list->device-vector)
         (only-in "python-env.rkt" run-twin))

(provide case-of
         on
         twin-results
         weights-of)

(define-runtime-path twin "../python/metrics_twin.py")

(define (twin-results cases)
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases cases)))
  (check-equal? version libcuml-version "T5")
  results)

(define (on xs dt)
  (if (list? (car xs))
      (list*->device-matrix xs #:dtype dt)
      (list->device-vector xs #:dtype dt)))

(define (weights-of w)
  (and w (list->device-vector w #:dtype 'float64)))

(define (case-of op y p w dt [pt dt])
  (hasheq 'op op
          'y_true y
          'y_pred p
          'dtype_true (symbol->string dt)
          'dtype_pred (symbol->string pt)
          'weight (or w (json-null))))
