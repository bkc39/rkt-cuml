#lang racket/base

(require (only-in "private/real-logistic-data.rkt" check-raw-optima labelled-set))

(check-raw-optima (labelled-set 'breast-cancer "diagnosis" '("id")))
