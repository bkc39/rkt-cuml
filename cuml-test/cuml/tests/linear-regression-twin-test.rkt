#lang racket/base

(require (only-in "private/linear-twin-data.rkt" check-cuml-failures check-cuml-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "float32: bitwise against Python cuML on the staged data, within 128 u kappa^2 on the raw data"
  (check-cuml-parity 'float32))

(test-twin "Python cuML gets a column at 1e-5, a y at 1e37 and a constant column under qr wrong; the fit does not"
  (check-cuml-failures))
