#lang racket/base

(require (only-in cuml/manifold spectral-embedding-fit)
         (only-in rackunit check-true)
         (only-in "private/blob-data.rkt" centred-blobs)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-data.rkt" graph-map-vals three-blocks unit-path)
         (only-in "private/spectral-embedding-data.rkt"
                  check-twins
                  column-gaps
                  cosine-gaps
                  embedded
                  embedded-graph
                  embedding-case
                  embedding-checks
                  graph-case
                  graph-checks
                  heavy-pair
                  quantised
                  rows-matrix
                  sign-ruled?
                  stochastic-graph)
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

(define (sklearn-graph name g options)
  (list name (graph-case g options #:op "sklearn") (near (embedded (embedded-graph g options)))))

(define (sklearn-dense name rows)
  (define result (spectral-embedding-fit (rows-matrix rows) #:n-neighbors 10))
  (list name (embedding-case rows result #:op "sklearn-dense") (near (embedded result))))

(define tiny-gap 0.1)

(define (sign-free-gaps ours theirs)
  (map min (column-gaps ours theirs) (column-gaps ours (map (lambda (row) (map - row)) theirs))))

(define ((departs ours #:gaps [gaps column-gaps]) theirs)
  (check-true (sign-ruled? theirs))
  (check-true (> (apply max (gaps ours theirs)) tiny-gap) (format "gaps ~a" (gaps ours theirs))))

(define (raw-tiny)
  (define result (spectral-embedding-fit (rows-matrix quantised) #:n-neighbors 10))
  (list "L7 Python cuML at 2^-70 X"
        (embedding-case quantised result #:op "raw-tiny")
        (departs (embedded result))))

(define heavy-options (hasheq 'norm-laplacian #f))

(define (heavy-departs)
  (list "L6 sklearn at a 2^16 pair without norm-laplacian"
        (graph-case (heavy-pair 16) heavy-options #:op "sklearn")
        (departs (embedded (embedded-graph (heavy-pair 16) heavy-options)))))

(define (path-departs)
  (define g (heavy-pair 16 #:graph (unit-path 24) #:between '(5 6)))
  (list "L6 sklearn at a 2^16 pair on a 24-node path without norm-laplacian, sign-free"
        (graph-case g heavy-options #:op "sklearn")
        (departs (embedded (embedded-graph g heavy-options)) #:gaps sign-free-gaps)))

(define (cuml-twins)
  (define blob-rows (fixture-rows blobs))
  (define digit-rows (fixture-rows (digits)))
  (define-values (blocks _truth) (three-blocks))
  (append (embedding-checks "L5 blobs" blob-rows (parity-configs 10))
          (embedding-checks "L5 blobs" blob-rows (parity-configs 256))
          (embedding-checks "L5 digits" digit-rows (parity-configs 10))
          (embedding-checks "L5 digits" digit-rows (parity-configs 256))
          (graph-checks "L5 sbm" (stochastic-graph) (parity-configs))
          (graph-checks "L5 sbm weighted" (stochastic-graph #:weighted? #t) (parity-configs))
          (graph-checks "L5 three-blocks" blocks (parity-configs))))

(define (sklearn-twins)
  (define g (stochastic-graph))
  (define-values (rows _labels) (centred-blobs '((0 0 0 0) (3 3 0 0) (0 3 3 0)) 70 0))
  (append
   (for/list ([seed (in-range 10)])
     (sklearn-graph (format "L6 sbm seed ~a" seed) (stochastic-graph '(40 60 80) seed) (hasheq)))
   (list
    (sklearn-graph "L6 weighted" (stochastic-graph #:weighted? #t) (hasheq))
    (sklearn-graph "L6 five blocks" (stochastic-graph '(30 40 50 60 70)) (hasheq 'n-components 4))
    (sklearn-graph "L6 drop-first #f" g (hasheq 'drop-first #f))
    (sklearn-graph "L6 norm-laplacian #f" g (hasheq 'norm-laplacian #f))
    (sklearn-graph "L6 norm-laplacian #f at 2^-20 weights"
                   (graph-map-vals g (lambda (w) (* w (expt 2.0 -20))))
                   (hasheq 'norm-laplacian #f))
    (sklearn-graph "L6 2^16 pair" (heavy-pair 16) (hasheq))
    (heavy-departs)
    (path-departs)
    (sklearn-dense "L6 overlapping blobs" rows))))

(test-twin "L5, L6, L7: fits are Python cuML's bitwise and sklearn's within 1e-4 but at 2^16 pairs without norm-laplacian; Python cuML on 2^-70 X is not"
  (check-twins (append (cuml-twins) (sklearn-twins) (list (raw-tiny)))))
