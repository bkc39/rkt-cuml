#lang racket/base

(require (only-in cuml
                  pca-fit
                  pca-result-explained-variance-ratio
                  pca-result-singular-values
                  table->device-matrix
                  tsvd-fit
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values)
         (only-in cuml/scribblings/guide/helpers features rounded)
         (only-in datasets/core load-dataset-table)
         (only-in racket/list first second take)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check check-equal? test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "private/guide-data.rkt" boundary-margin host scaled-matrix)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/scaler-data.rkt" [float32-roundoff u]))

(define-runtime-path twin "python/guide_twin.py")

(define (twin-case op samples k)
  (hasheq 'op op 'rows (device-matrix->list* samples) 'k k))

(define (check-printed name ours theirs bound factor)
  (test-case name
    (check-equal? (length theirs) (length ours))
    (check-equal? (length bound) (length ours))
    (for ([value (in-list ours)]
          [reference (in-list theirs)]
          [b (in-list bound)])
      (check >= (boundary-margin reference 3) (* factor b))
      (check-equal? (rounded value) (rounded reference)))))

(test-twin "the decomposition chapter prints sklearn's values, each far from a rounding boundary"
  (define measurements (features 'wine "cultivar"))
  (define raw (table->device-matrix (load-dataset-table 'wine) #:columns measurements))
  (define scaled (scaled-matrix 'wine measurements))
  (match-define (hash-table ['results
                             (list (hash-table ['ratio raw-ratio])
                                   (hash-table ['ratio ratio]
                                               ['singular_values singular]
                                               ['explained_variance variance])
                                   (hash-table ['ratio svd-ratio] ['singular_values svd-singular]))]
                            ['version version])
    (run-twin twin
              (hasheq 'cases (list (twin-case "pca" raw 2)
                                   (twin-case "pca" scaled 13)
                                   (twin-case "tsvd" scaled 2)))))
  (check-equal? version "1.8.0")
  (define raw-fit (pca-fit raw #:n-components 2))
  (define scaled-fit (pca-fit scaled))
  (define svd-fit (tsvd-fit scaled #:n-components 2))
  (define ratio-bound (* 64 u (first ratio)))
  (define (singular-bound sigma)
    (/ (* (sub1 (length (device-matrix->list* scaled))) 64 u (first variance)) (* 2 sigma)))
  (define pca-sigma-bounds (map singular-bound (take singular 2)))
  (check-printed "raw ratios"
                 (host (pca-result-explained-variance-ratio raw-fit) 2)
                 raw-ratio
                 (list (* 64 u (first raw-ratio)) (* 64 u (first raw-ratio)))
                 10)
  (check-printed "scaled ratios"
                 (host (pca-result-explained-variance-ratio scaled-fit) 3)
                 (take ratio 3)
                 (list ratio-bound ratio-bound ratio-bound)
                 10)
  (check-printed "two components' share"
                 (list (apply + (host (pca-result-explained-variance-ratio scaled-fit) 2)))
                 (list (+ (first ratio) (second ratio)))
                 (list (* 2 ratio-bound))
                 10)
  (check-printed "PCA's ratios"
                 (host (pca-result-explained-variance-ratio scaled-fit) 2)
                 (take ratio 2)
                 (list ratio-bound ratio-bound)
                 10)
  (check-printed "truncated SVD's ratios"
                 (host (tsvd-result-explained-variance-ratio svd-fit) 2)
                 svd-ratio
                 (list (+ ratio-bound (* 64 u)) (+ ratio-bound (* 64 u)))
                 10)
  (check-printed "PCA's singular values"
                 (host (pca-result-singular-values scaled-fit) 2)
                 (take singular 2)
                 pca-sigma-bounds
                 1)
  (check-printed "truncated SVD's singular values"
                 (host (tsvd-result-singular-values svd-fit) 2)
                 svd-singular
                 (map (lambda (b) (+ b (* 64 u (first singular)))) pca-sigma-bounds)
                 1))
