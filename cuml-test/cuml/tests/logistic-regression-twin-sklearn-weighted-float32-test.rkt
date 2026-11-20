#lang racket/base

(require (only-in racket/list filter-not)
         (only-in "private/logistic-data.rkt" blob-case penalty-settings setting-penalty)
         (only-in "private/logistic-sklearn-twin.rkt" check-sklearn-twins sklearn-trials)
         (only-in "private/python-env.rkt" test-twin))

(define cases
  (list (blob-case "three classes weighted" 120 3 3 34 #:weighted? #t)
        (blob-case "five classes weighted" 150 3 5 35 #:weighted? #t)))

(define penalised (filter-not (lambda (s) (eq? (setting-penalty s) 'none)) penalty-settings))

(test-twin "float32: weighted multiclass fits are as near sklearn's optimum as cuML's fit of the repeated rows"
  (check-sklearn-twins (sklearn-trials cases penalised #:dtypes '(float32)) #:repeated? #t))
