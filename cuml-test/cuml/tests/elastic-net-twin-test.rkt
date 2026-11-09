#lang racket/base

(require (only-in "private/elastic-net-twin-data.rkt" check-raw-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "float32: unweighted and dyadic-weight fits are Python cuML's ElasticNet bit for bit, both selections (E10, E11)"
  (check-raw-parity 'float32))
