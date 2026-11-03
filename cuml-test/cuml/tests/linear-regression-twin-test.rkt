#lang racket/base

(require (only-in "private/linear-twin-data.rkt" check-cuml-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "P1, P2 float32: bitwise against Python cuML on the staged data, within 128 u kappa^2 on the raw data"
  (check-cuml-parity 'float32))
