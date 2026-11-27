#lang racket/base

(require (only-in "private/svm-twin-data.rkt" check-sklearn-dtype))

(check-sklearn-dtype 'float64)
