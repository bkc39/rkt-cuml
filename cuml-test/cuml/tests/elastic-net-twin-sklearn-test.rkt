#lang racket/base

(require (only-in "private/elastic-net-twin-data.rkt" check-sklearn-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "fits are within sklearn-distance of scikit-learn's"
  (check-sklearn-parity 'float32)
  (check-sklearn-parity 'float64))
