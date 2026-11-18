#lang racket/base

(require (only-in racket/list filter-not)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in "private/logistic-data.rkt" fit-flat fit-parts relative-gap)
         (only-in "private/logistic-twin-data.rkt" logistic-twin-results their-fit twin-trial)
         (only-in "private/logistic-twin-rows.rkt" repeatable? request row-fit row-name rows)
         (only-in "private/python-env.rkt" test-twin))

(define owl-qn-gap-bound 1e-3)

(test-twin "float32 multiclass OWL-QN fits, which repeat on neither side, are near Python cuML's"
  (define kept (filter-not repeatable? rows))
  (for ([r (in-list kept)]
        [theirs (in-list (logistic-twin-results (map (lambda (k) (request k #:op "cuml-fit"))
                                                     kept)))])
    (match-define (twin-trial dt _ _ _) r)
    (test-case (row-name r)
      (define gap (relative-gap (fit-flat (fit-parts (row-fit r))) (fit-flat (their-fit theirs dt))))
      (check-true (<= gap owl-qn-gap-bound) (format "gap ~a" gap)))))
