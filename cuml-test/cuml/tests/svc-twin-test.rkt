#lang racket/base

(require (only-in cuml/svm svc-params)
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/svm-twin-data.rkt" svc-twin-params))

(test-twin "the hyperparameters are Python cuML's, with max-iter #f for its -1"
  (define theirs
    (svc-twin-params #:unbound '(class-weight decision-function-shape random-state epsilon)))
  (check-equal? (hash-ref theirs 'max-iter) -1)
  (check-equal? (hash-ref (svc-params) 'max-iter) #f)
  (check-equal? (hash-set theirs 'max-iter #f) (svc-params)))
