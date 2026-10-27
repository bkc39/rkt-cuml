#lang racket/base

(require (only-in cuml/manifold spectral-embedding-fit)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in "private/python-env.rkt" run-cases test-twin)
         (only-in "private/spectral-embedding-data.rkt"
                  column-gaps
                  cosine-gaps
                  embedded
                  embedded-graph
                  graph-case
                  rows-matrix
                  sbm
                  se-case
                  sign-ruled?
                  twin)
         (only-in "private/umap-data.rkt" blobs fixture-rows))

(define gap-bound 1e-4)

(define cosine-bound 1e-9)

(define (check-near name ours theirs)
  (test-case name
    (check-true (sign-ruled? ours))
    (check-true (<= (apply max (column-gaps ours theirs)) gap-bound)
                (format "column gaps ~a" (column-gaps ours theirs)))
    (check-true (<= (apply max (cosine-gaps ours theirs)) cosine-bound)
                (format "cosine gaps ~a" (cosine-gaps ours theirs)))))

(define (check-graphs entries)
  (define theirs
    (run-cases twin
               (for/list ([entry (in-list entries)])
                 (match-define (list _ g options) entry)
                 (graph-case g options #:op "sklearn"))))
  (for ([entry (in-list entries)]
        [found (in-list theirs)])
    (match-define (list name g options) entry)
    (check-near name (embedded (embedded-graph g options)) (hash-ref found 'embedding))))

(define (planted sizes seed #:weighted? [weighted? #f])
  (define-values (g _truth) (sbm sizes 0.3 0.01 seed #:weighted? weighted?))
  g)

(test-twin "L6: graph fits on planted blocks are sklearn's within 1e-4, for ten seeds"
  (check-graphs (for/list ([seed (in-range 10)])
                  (list (format "sbm seed ~a" seed) (planted '(40 60 80) seed) (hasheq)))))

(test-twin "L6: weighted, five-block, drop-first #f and norm-laplacian #f fits are sklearn's within 1e-4"
  (define g (planted '(40 60 80) 0))
  (check-graphs (list (list "weighted" (planted '(40 60 80) 0 #:weighted? #t) (hasheq))
                      (list "five blocks" (planted '(30 40 50 60 70) 0) (hasheq 'n-components 4))
                      (list "drop-first #f" g (hasheq 'drop-first #f))
                      (list "norm-laplacian #f" g (hasheq 'norm-laplacian #f)))))

(test-twin "L6: a dense fit on blobs at 10 neighbours is sklearn's within 1e-4"
  (define rows (fixture-rows blobs))
  (define result (spectral-embedding-fit (rows-matrix rows) #:n-neighbors 10))
  (define theirs (run-cases twin (list (se-case rows result #:op "sklearn-dense"))))
  (check-near "blobs" (embedded result) (hash-ref (car theirs) 'embedding)))
