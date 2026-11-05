#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/ridge-twin-data.rkt" check-sklearn-parity))

(test-twin "sklearn Ridge svd and cholesky, within 256 u kappa_alpha, both solvers, both dtypes"
  (check-sklearn-parity 'float32)
  (check-sklearn-parity 'float64))
