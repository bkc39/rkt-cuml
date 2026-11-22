#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/svm-twin-data.rkt"
                  breast-cancer
                  check-sklearn-twins
                  separable-blobs
                  sklearn-trials))

(test-twin "standardised breast-cancer float64 decisions are near sklearn's, labels equal off the boundary"
  (check-sklearn-twins (sklearn-trials (breast-cancer) 'float64)))

(test-twin "separable float64 blobs give sklearn's labels; the support sets differ only on the margin"
  (check-sklearn-twins (sklearn-trials (separable-blobs) 'float64) #:separable? #t))
