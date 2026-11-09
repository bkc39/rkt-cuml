#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-predict)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y normal-case on-matrix on-vector)
         (only-in "private/process-memory.rkt" test-fit-memory))

(define (setup)
  (define c (normal-case 2000 7 0 #:weighted? #t))
  (define mixed (normal-case 60 4 11))
  (define draw (uniform-source 3))
  (define X (on-matrix (lcase-X c) 'float32))
  (define y (on-vector (lcase-y c) 'float32))
  (define w (on-vector (lcase-w c) 'float32))
  (define wide
    (on-matrix (for/list ([_ (in-range 2)])
                 (for/list ([_ (in-range 24000)])
                   (draw)))
               'float32))
  (define wide-y (on-vector '(1 2) 'float32))
  (define mixed-X
    (on-matrix (for/list ([row (in-list (lcase-X mixed))])
                 (map * row (list (expt 2 -13) 1 (expt 2 13) 1)))
               'float32))
  (define mixed-y (on-vector (lcase-y mixed) 'float32))
  (values (lambda () (ridge-predict (ridge-fit X y #:sample-weight w #:alpha 1/2 #:solver 'svd) X))
          (list (list "ridge-fit: solver: expected auto, eig or svd"
                      (lambda () (ridge-fit X y #:solver 'lsmr)))
                (list "ridge-fit: alpha: expected a finite value at least 0, got -1"
                      (lambda () (ridge-fit X y #:alpha -1)))
                (list "ridge-fit: X: 2x24000 elements are past cuSOLVER's eig workspace query"
                      (lambda () (ridge-fit wide wide-y)))
                (list #rx"^ridge-fit: X: the eig solve is not a ridge solution "
                      (lambda () (ridge-fit mixed-X mixed-y #:alpha 1 #:solver 'eig))))))

(test-fit-memory "ridge" #:rounds 200 #:sample-every 20 #:setup setup)
