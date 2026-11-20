#lang racket/base

(require (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" run-twin test-python))

(define-runtime-path script "python/params_of_twin.py")

(define params (list (hasheq 'op "params")))

(define rows
  `(("params_of under run(fixtures=True), fixtures given" ,(hasheq 'cases params 'fixtures (hasheq)))
    ("params_of under run(fixtures=True), no fixtures" ,(hasheq 'cases params))))

(for ([row (in-list rows)])
  (match-define (list name input) row)
  (test-python name
    (check-equal? (hash-ref (run-twin script input) 'results)
                  (list (hasheq 'alpha 1.5 'solver "auto")))))
