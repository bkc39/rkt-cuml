#lang racket/base

(require (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-fit-transform
                  spectral-embedding-params
                  spectral-embedding-result-embedding
                  spectral-embedding-result-n-neighbors
                  spectral-embedding-result-params
                  spectral-embedding-result?)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape)
         (only-in "private/spectral-data.rkt" graph->vectors graph-n three-blocks)
         (only-in "private/spectral-embedding-data.rkt" embedded embedded-graph rows-matrix)
         (only-in "private/umap-data.rkt" blobs far-blobs finite-rows? fixture-rows fixture-X))

(define-values (blocks _truth) (three-blocks))

(define geometric
  (rows-matrix (for/list ([i (in-range 25)])
                 (list (expt 1.25 i) 0.0))))

(test-gpu "L1: the embedding is a fresh n×k float32 matrix, with the neighbours used and the params"
  (for* ([k (in-list '(1 2 5))]
         [drop (in-list '(#t #f))]
         [norm (in-list '(#t #f))])
    (test-case (format "k ~a drop ~a norm ~a" k drop norm)
      (define X (fixture-X far-blobs))
      (define result
        (spectral-embedding-fit X #:n-components k #:drop-first drop #:norm-laplacian norm))
      (check-true (spectral-embedding-result? result))
      (check-equal? (matrix-shape (spectral-embedding-result-embedding result))
                    (list #t 'float32 'row-major (list 300 k)))
      (check-false (eq? (spectral-embedding-result-embedding result) X))
      (check-true (finite-rows? (embedded result)))
      (check-equal? (spectral-embedding-result-n-neighbors result) 30)
      (check-equal? (spectral-embedding-result-params result)
                    (spectral-embedding-params #:n-components k
                                               #:drop-first drop
                                               #:norm-laplacian norm)))))

(test-gpu "L1: the default neighbour count is min(256, max(2, ⌊n/10⌋)), and an explicit one wins"
  (for ([entry (in-list (list (list geometric #f 2)
                              (list (fixture-X far-blobs) #f 30)
                              (list (fixture-X blobs) #f 100)
                              (list (fixture-X far-blobs) 7 7)))])
    (define-values (X given used) (apply values entry))
    (define result (spectral-embedding-fit X #:n-neighbors given #:n-components 1))
    (check-equal? (spectral-embedding-result-n-neighbors result) used)
    (check-equal? (hash-ref (spectral-embedding-result-params result) 'n-neighbors) given)))

(test-gpu "L1: a graph fit reports no neighbour count, and fit-transform returns its embedding"
  (define result (embedded-graph blocks))
  (check-equal? (spectral-embedding-result-n-neighbors result) #f)
  (check-equal? (matrix-shape (spectral-embedding-result-embedding result))
                (list #t 'float32 'row-major (list (graph-n blocks) 2)))
  (check-equal? (spectral-embedding-result-params result) (spectral-embedding-params))
  (define-values (fitted matrix) (spectral-embedding-fit-transform (fixture-X far-blobs)))
  (check-true (eq? matrix (spectral-embedding-result-embedding fitted))))

(test-gpu "L1: params keep a derived n-neighbors and a drawn seed as #f"
  (define params
    (spectral-embedding-result-params (spectral-embedding-fit (fixture-X far-blobs)
                                                              #:random-state #f)))
  (check-equal? (hash-ref params 'n-neighbors) #f)
  (check-equal? (hash-ref params 'random-state) #f))

(test-gpu "L2: the fits leave X and the graph bitwise unchanged, on the rescale path too"
  (for ([rows (in-list (list (fixture-rows far-blobs)
                             (for/list ([row (in-list (fixture-rows far-blobs))])
                               (for/list ([x (in-list row)])
                                 (* x (expt 2.0 -70))))))])
    (define X (rows-matrix rows))
    (define before (device-matrix->list* X))
    (spectral-embedding-fit X #:n-neighbors 10)
    (check-true (bits-equal? (device-matrix->list* X) before)))
  (define vectors (graph->vectors blocks))
  (define before (map device-vector->list vectors))
  (apply spectral-embedding-fit-graph (append vectors (list (graph-n blocks))))
  (check-equal? (map device-vector->list vectors) before))
