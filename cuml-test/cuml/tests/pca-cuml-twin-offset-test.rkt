#lang racket/base

(require (only-in "private/pca-twin-data.rkt" test-against-cuml))

(for-each test-against-cuml '("offset" "low rank"))
