#lang racket/base

(require (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-fit-transform)
         (only-in racket/list make-list)
         (only-in "private/process-memory.rkt" test-rounds-and-refusals)
         (only-in "private/scaler-data.rkt" normal-rows rows->matrix)
         (only-in "private/spectral-data.rkt"
                  edited-vectors
                  graph->vectors
                  graph-n
                  graph-rows
                  set-entry
                  sparse-blocks))

(define blocks (sparse-blocks '(300 400 500) 0.05 0))

(define vectors (graph->vectors blocks))

(define (fit-graph vectors . options)
  (apply spectral-embedding-fit-graph (append vectors (list (graph-n blocks)) options)))

(define (one-round X)
  (spectral-embedding-fit X #:n-components 8)
  (spectral-embedding-fit-transform X #:n-neighbors 64)
  (fit-graph vectors #:n-components 4))

(define (refusals _fitted)
  (define nan-row (rows->matrix (cons (list +nan.0 1.0) (make-list 99 '(1.0 2.0))) 'float32))
  (define halves (rows->matrix (append (make-list 100 '(1.0 2.0)) (make-list 100 '(3.0 5.0))) 'float32))
  (define few (rows->matrix (for/list ([i (in-range 5)]) (list (exact->inexact i) 1.0)) 'float32))
  (define looped
    (edited-vectors blocks #:cols (set-entry 3 (list-ref (graph-rows blocks) 3))))
  (list (lambda () (spectral-embedding-fit nan-row))
        (lambda () (spectral-embedding-fit halves #:n-neighbors 10))
        (lambda () (spectral-embedding-fit few #:n-neighbors 2))
        (lambda () (spectral-embedding-fit-transform few #:n-components 0))
        (lambda () (fit-graph looped))))

(test-rounds-and-refusals "fit, fit-transform and fit-graph"
                          (lambda () (rows->matrix (normal-rows 7 4096 16 'float32) 'float32))
                          one-round
                          refusals
                          #:rounds 30
                          #:buffers-per-round 3)
