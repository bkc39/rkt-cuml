#lang racket/base

(require (only-in racket/list filter-not)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in "private/logistic-data.rkt" fit-flat fit-parts relative-gap)
         (only-in "private/logistic-twin-data.rkt" their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt"
                  for-each-twin
                  repeatable?
                  request
                  row-fit
                  row-name
                  rows)
         (only-in "private/python-env.rkt" test-twin))

(define owl-qn-gap-bound 1e-3)

(test-twin "float32 multiclass OWL-QN: near Python cuML's"
  (for-each-twin
   (lambda (r theirs)
     (match-define (struct* twin-trial ([dt dt])) r)
     (test-case (row-name r)
       (define gap (relative-gap (fit-flat (fit-parts (row-fit r))) (fit-flat (their-fit theirs dt))))
       (check-true (<= gap owl-qn-gap-bound) (format "gap ~a" gap))))
   (filter-not repeatable? rows)
   #:request (lambda (k) (request k #:op "cuml-fit"))))
