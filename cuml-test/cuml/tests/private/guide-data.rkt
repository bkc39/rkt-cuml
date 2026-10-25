#lang racket/base

(require (only-in cuml standard-scaler-fit-transform table->device-matrix)
         (only-in datasets/core load-dataset-table)
         (only-in threading ~>))

(provide scaled-matrix)

(define (scaled-matrix name columns)
  (define-values (_fitted scaled)
    (~> (load-dataset-table name)
        (table->device-matrix #:columns columns)
        standard-scaler-fit-transform))
  scaled)
