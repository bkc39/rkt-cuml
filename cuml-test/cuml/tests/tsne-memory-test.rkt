#lang racket/base

(require (only-in cuml/manifold tsne-fit tsne-fit-transform)
         (only-in racket/list make-list)
         (only-in raft list*->device-matrix)
         (only-in "private/common-data.rkt" blank-matrix)
         (only-in "private/process-memory.rkt" test-rounds-and-refusals)
         (only-in "private/scaler-data.rkt" normal-rows)
         (only-in "private/umap-data.rkt" fixture-X tiny))

(define iterations 250)

(define (one-round X)
  (tsne-fit-transform X #:max-iter iterations)
  (tsne-fit X #:max-iter iterations))

(define (refusals _fitted)
  (define nan-row
    (list*->device-matrix (cons (list +nan.0 1.0) (make-list 99 '(1.0 2.0))) #:dtype 'float32))
  (define constant (list*->device-matrix (make-list 100 '(1.0 2.0)) #:dtype 'float32))
  (define narrow (blank-matrix 40 1 'float32))
  (define one-row (blank-matrix 1 16 'float32))
  (define tiny-X (fixture-X tiny))
  (list (lambda () (tsne-fit nan-row))
        (lambda () (tsne-fit constant))
        (lambda () (tsne-fit one-row))
        (lambda () (tsne-fit narrow))
        (lambda () (tsne-fit-transform constant #:perplexity 0.5))
        (lambda () (tsne-fit tiny-X #:perplexity 5.0 #:learning-rate 1e30))))

(test-rounds-and-refusals "fit and fit-transform"
                          (lambda ()
                            (list*->device-matrix (normal-rows 7 4096 16 'float32) #:dtype 'float32))
                          one-round
                          refusals
                          #:rounds 30
                          #:buffers-per-round 2)
