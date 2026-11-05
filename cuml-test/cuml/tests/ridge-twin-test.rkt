#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/ridge-twin-data.rkt" check-raw-parity check-staged-parity))

(test-twin "float32: on already-staged data the eig fit is Python cuML's Ridge(solver='eig'), bitwise"
  (check-staged-parity 'float32))

(test-twin "float32: raw Python cuML Ridge, eig and svd, within 256 u kappa_alpha"
  (check-raw-parity 'float32))
