#lang racket/base

(require (only-in "private/pca-twin-data.rkt" test-against-cuml))

(test-against-cuml "scales")
