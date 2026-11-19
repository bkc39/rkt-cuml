#lang racket/base

(require (only-in "real-logistic-data.rkt" labelled-set))

(provide raw-breast-cancer)

(define raw-breast-cancer (labelled-set 'breast-cancer "diagnosis" '("id")))
