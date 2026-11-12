#lang racket/base

(require (only-in "private/elastic-net-twin-data.rkt" check-raw-parity)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "float32 fits are Python cuML's ElasticNet bit for bit"
  (check-raw-parity 'float32))
