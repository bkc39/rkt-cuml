#lang racket/base

(require (only-in cuml/manifold umap-fit umap-fit-transform umap-transform)
         (only-in racket/list make-list)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/process-memory.rkt" test-rounds-and-refusals)
         (only-in "private/scaler-data.rkt" normal-rows rows->matrix))

(define (one-round X)
  (define fitted (umap-fit X #:init 'random #:n-epochs 30))
  (umap-transform fitted X)
  (umap-fit-transform X #:init 'random #:n-epochs 30)
  fitted)

(define (refusals fitted)
  (define nan-row (rows->matrix (cons (list +nan.0 1.0) (make-list 99 '(1.0 2.0))) 'float32))
  (define narrow (blank-matrix 4 3 'float32))
  (define no-rows (blank-matrix 0 16 'float32))
  (define one-row (blank-matrix 1 16 'float32))
  (list (lambda () (umap-fit nan-row))
        (lambda () (umap-fit one-row))
        (lambda () (umap-fit narrow #:min-dist 2.0))
        (lambda () (umap-transform fitted narrow))
        (lambda () (umap-transform fitted no-rows))))

(test-rounds-and-refusals "fit, transform and fit-transform"
                          (lambda () (rows->matrix (normal-rows 7 4096 16 'float32) 'float32))
                          one-round
                          refusals
                          #:rounds 30
                          #:buffers-per-round 3)
