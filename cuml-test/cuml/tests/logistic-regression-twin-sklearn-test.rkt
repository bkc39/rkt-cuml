#lang racket/base

(require (only-in "private/logistic-data.rkt" blob-case penalty-settings)
         (only-in "private/logistic-sklearn-twin.rkt" check-sklearn-twins sklearn-trials)
         (only-in "private/python-env.rkt" test-twin))

(define cases
  (list (blob-case "binary" 100 3 2 31)
        (blob-case "binary weighted" 100 3 2 32 #:weighted? #t)
        (blob-case "three classes" 120 3 3 33)))

(test-twin "the fit is near sklearn's optimum and predicts its labels off the decision boundary"
  (check-sklearn-twins (sklearn-trials cases penalty-settings)))
