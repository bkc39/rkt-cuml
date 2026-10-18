#lang racket/base

(require (only-in cuml/estimator clusterer? estimator? predictor? scorer? transformer?))

(provide capabilities)

(define (capabilities v . more)
  (map (lambda (p) (p v)) (list* estimator? predictor? clusterer? transformer? scorer? more)))
