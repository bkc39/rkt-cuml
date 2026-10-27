#lang racket/base

(require (only-in cuml/manifold spectral-embedding-fit)
         (only-in rackunit check-true)
         (only-in "private/blob-data.rkt" centred-blobs)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-data.rkt" three-blocks)
         (only-in "private/spectral-embedding-data.rkt"
                  check-twins
                  column-gaps
                  cosine-gaps
                  embedded
                  embedded-graph
                  graph-case
                  graph-checks
                  rows-matrix
                  sbm
                  se-case
                  se-checks
                  sign-ruled?)
         (only-in "private/spectral-embedding-twin-data.rkt" parity-configs)
         (only-in "private/umap-data.rkt" blobs digits fixture-rows))

(define gap-bound 1e-4)

(define cosine-bound 1e-9)

(define ((near ours) theirs)
  (check-true (sign-ruled? ours))
  (check-true (<= (apply max (column-gaps ours theirs)) gap-bound)
              (format "column gaps ~a" (column-gaps ours theirs)))
  (check-true (<= (apply max (cosine-gaps ours theirs)) cosine-bound)
              (format "cosine gaps ~a" (cosine-gaps ours theirs))))

(define (planted sizes seed #:weighted? [weighted? #f])
  (define-values (g _truth) (sbm sizes 0.3 0.01 seed #:weighted? weighted?))
  g)

(define (sklearn-graph name g options)
  (list name (graph-case g options #:op "sklearn") (near (embedded (embedded-graph g options)))))

(define (sklearn-dense name rows)
  (define result (spectral-embedding-fit (rows-matrix rows) #:n-neighbors 10))
  (list name (se-case rows result #:op "sklearn-dense") (near (embedded result))))

(define (cuml-twins)
  (define blob-rows (fixture-rows blobs))
  (define digit-rows (fixture-rows (digits)))
  (define-values (blocks _truth) (three-blocks))
  (append (se-checks "L5 blobs" blob-rows (parity-configs 10))
          (se-checks "L5 blobs" blob-rows (parity-configs 256))
          (se-checks "L5 digits" digit-rows (parity-configs 10))
          (se-checks "L5 digits" digit-rows (parity-configs 256))
          (graph-checks "L5 sbm" (planted '(40 60 80) 0) (parity-configs))
          (graph-checks "L5 sbm weighted" (planted '(40 60 80) 0 #:weighted? #t) (parity-configs))
          (graph-checks "L5 three-blocks" blocks (parity-configs))))

(define (sklearn-twins)
  (define g (planted '(40 60 80) 0))
  (define-values (rows _labels) (centred-blobs '((0 0 0 0) (3 3 0 0) (0 3 3 0)) 70 0))
  (append
   (for/list ([seed (in-range 10)])
     (sklearn-graph (format "L6 sbm seed ~a" seed) (planted '(40 60 80) seed) (hasheq)))
   (list (sklearn-graph "L6 weighted" (planted '(40 60 80) 0 #:weighted? #t) (hasheq))
         (sklearn-graph "L6 five blocks" (planted '(30 40 50 60 70) 0) (hasheq 'n-components 4))
         (sklearn-graph "L6 drop-first #f" g (hasheq 'drop-first #f))
         (sklearn-graph "L6 norm-laplacian #f" g (hasheq 'norm-laplacian #f))
         (sklearn-dense "L6 overlapping blobs" rows))))

(test-twin "L5, L6: fits on blobs, digits and planted graphs are Python cuML's bitwise, and sklearn's within 1e-4"
  (check-twins (append (cuml-twins) (sklearn-twins))))
