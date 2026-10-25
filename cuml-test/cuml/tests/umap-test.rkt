#lang racket/base

(require (only-in cuml/manifold
                  umap-fit
                  umap-fit-transform
                  umap-params
                  umap-result-X
                  umap-result-a
                  umap-result-b
                  umap-result-embedding
                  umap-result-params
                  umap-result?
                  umap-transform)
         (only-in racket/flonum flsingle)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape)
         (only-in "private/umap-data.rkt" far-blobs fixture-X fixture-rows fixture-y overlap))

(define (single? x)
  (and (flonum? x) (eqv? x (flsingle x))))

(test-gpu "L1: the embedding is a fresh n×k float32 matrix, and the result keeps X and the params"
  (for* ([init (in-list '(spectral random))]
         [k (in-list '(1 2 3))])
    (test-case (format "~a k=~a" init k)
      (define X (fixture-X far-blobs))
      (define result (umap-fit X #:init init #:n-components k))
      (check-true (umap-result? result))
      (check-equal? (matrix-shape (umap-result-embedding result))
                    (list #t 'float32 'row-major (list 300 k)))
      (check-true (eq? (umap-result-X result) X))
      (check-false (eq? (umap-result-embedding result) X))
      (check-true (single? (umap-result-a result)))
      (check-true (single? (umap-result-b result)))
      (check-equal? (umap-result-params result) (umap-params #:init init #:n-components k))
      (define T (umap-transform result X))
      (check-equal? (matrix-shape T) (list #t 'float32 'row-major (list 300 k)))
      (check-false (eq? T X)))))

(test-gpu "L1: explicit curve parameters are stored as their float32 values"
  (define result (umap-fit (fixture-X far-blobs) #:a 1.1 #:b 0.7))
  (check-equal? (umap-result-a result) (flsingle 1.1))
  (check-equal? (umap-result-b result) (flsingle 0.7))
  (check-equal? (hash-ref (umap-result-params result) 'a) 1.1))

(test-gpu "L1: fit-transform's matrix is its result's embedding"
  (define-values (result E) (umap-fit-transform (fixture-X far-blobs)))
  (check-true (eq? E (umap-result-embedding result))))

(test-gpu "L2: fit leaves X and y bitwise unchanged; transform leaves the result unchanged"
  (define X (fixture-X overlap))
  (define y (fixture-y overlap 'int64))
  (define result (umap-fit X #:y y #:init 'random))
  (check-true (bits-equal? (device-matrix->list* X) (fixture-rows overlap)) "X")
  (check-equal? (device-vector->list y) (device-vector->list (fixture-y overlap 'int64)) "y")
  (define before (device-matrix->list* (umap-result-embedding result)))
  (umap-transform result X)
  (check-true (bits-equal? (device-matrix->list* X) (fixture-rows overlap)) "result X")
  (check-true (bits-equal? (device-matrix->list* (umap-result-embedding result)) before) "embedding"))
