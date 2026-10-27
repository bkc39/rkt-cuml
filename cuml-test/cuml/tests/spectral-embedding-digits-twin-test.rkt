#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-embedding-data.rkt" check-se-parity)
         (only-in "private/spectral-embedding-twin-data.rkt" parity-configs)
         (only-in "private/umap-data.rkt" digits fixture-rows))

(test-twin "L5: dense fits on digits are Python cuML's, bitwise, at 10 and 256 neighbours"
  (define rows (fixture-rows (digits)))
  (for ([n-neighbors (in-list '(10 256))])
    (check-se-parity "digits" rows (parity-configs n-neighbors))))
