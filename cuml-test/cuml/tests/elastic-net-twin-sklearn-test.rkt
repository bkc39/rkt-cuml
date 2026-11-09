#lang racket/base

(require (only-in "private/elastic-net-twin-data.rkt" check-sklearn-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "sklearn ElasticNet and Lasso within tol max|c| sum sigma |sigma| / mu, both dtypes (S4)"
  (check-sklearn-parity 'float32)
  (check-sklearn-parity 'float64))
