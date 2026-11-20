#lang racket/base

(require (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" run-twin test-python))

(define-runtime-path params-of "python/params_of_twin.py")

(define-runtime-path versions-of "python/versions_of_twin.py")

(define params (list (hasheq 'op "params")))

(define rows
  `(("params_of under run(fixtures=True), fixtures given" ,params-of
                                                          ,(hasheq 'cases params 'fixtures (hasheq))
                                                          #f)
    ("params_of under run(fixtures=True), no fixtures" ,params-of ,(hasheq 'cases params) #f)
    ("run(versions_of=(\"numpy\",)) records numpy's version" ,versions-of
                                                             ,(hasheq 'cases params)
                                                             #t)))

(for ([row (in-list rows)])
  (match-define (list name script input numpy?) row)
  (test-python name
    (match-define (hash-table ('results results) ('version version)) (run-twin script input))
    (check-equal? results (list (hasheq 'alpha 1.5 'solver "auto")))
    (check-equal? (string? (hash-ref version 'numpy #f)) numpy? "version.numpy")))

(define-runtime-path put-of "python/put_twin.py")

(define bits '(a_bits 1069547520 3221225472))

(define put-rows
  `(("put writes the bits alone" "put" "float32" (1.5 -2.0) (,bits))
    ("put(float64=True) writes the float64 list beside the bits" "put_float64"
                                                                 "float32"
                                                                 (1.5 -2.0)
                                                                 ((a 1.5 -2.0) ,bits))
    ("put(float64=True) keeps a float64 value's own bits" "put_float64"
                                                          "float64"
                                                          (0.1)
                                                          ((a 0.1) (a_bits 4591870180066957722)))))

(for ([row (in-list put-rows)])
  (match-define (list name op dtype values stored) row)
  (test-python name
    (match-define (hash-table ('results results))
      (run-twin put-of (hasheq 'cases (list (hasheq 'op op 'values values 'dtype dtype)))))
    (check-equal? results (list (make-immutable-hasheq stored)))))
