#lang racket/base

(require (only-in cuml/cluster
                  hdbscan-approximate-predict
                  hdbscan-condensed-tree-lambdas
                  hdbscan-condensed-tree-selected-clusters
                  hdbscan-fit
                  hdbscan-linkage-tree-children
                  hdbscan-linkage-tree-distances
                  hdbscan-linkage-tree-sizes
                  hdbscan-prediction-labels
                  hdbscan-result-cluster-persistence
                  hdbscan-result-condensed-tree
                  hdbscan-result-labels
                  hdbscan-result-minimum-spanning-tree
                  hdbscan-result-probabilities
                  hdbscan-result-single-linkage-tree
                  hdbscan-spanning-tree-sources
                  hdbscan-spanning-tree-weights)
         (only-in racket/list count make-list range shuffle take)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in raft device-matrix->list* device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt" mat pick same-partition? seeded)
         (only-in "private/hdbscan-data.rkt"
                  dataset
                  dataset-names
                  descent-problems
                  dup
                  fitted-labels
                  fitted-lambdas
                  fitted-of
                  fitted-persistence
                  fitted-probabilities
                  label-count
                  structure-problems))

(define flt-max 3.4028234663852886e38)

(test-gpu "HD1: outputs have their dtypes, shapes and structure"
  (define result (hdbscan-fit (mat (dataset 'b50) 'float32)))
  (define tree (hdbscan-result-condensed-tree result))
  (define linkage (hdbscan-result-single-linkage-tree result))
  (define mst (hdbscan-result-minimum-spanning-tree result))
  (define k (length (device-vector->list (hdbscan-condensed-tree-selected-clusters tree))))
  (check-equal? (map (lambda (a) (list (dtype a) (shape a)))
                     (list (hdbscan-result-labels result)
                           (hdbscan-result-probabilities result)
                           (hdbscan-result-cluster-persistence result)
                           (hdbscan-linkage-tree-children linkage)
                           (hdbscan-linkage-tree-distances linkage)
                           (hdbscan-linkage-tree-sizes linkage)
                           (hdbscan-spanning-tree-sources mst)
                           (hdbscan-spanning-tree-weights mst)))
                (list (list 'int64 '(300))
                      (list 'float32 '(300))
                      (list 'float32 (list k))
                      (list 'int64 '(299 2))
                      (list 'float32 '(299))
                      (list 'int64 '(299))
                      (list 'int64 '(299))
                      (list 'float32 '(299))))
  (for ([name (in-list dataset-names)])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (check-equal? (structure-problems (fitted-of rows) (length rows)) '()))))

(test-gpu "HD2: every point descends from its selected cluster, and no noise point does"
  (for ([name (in-list dataset-names)])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (check-equal? (descent-problems (fitted-of rows) (length rows)) '()))))

(test-gpu "HD3: min-samples #f is min-samples = min-cluster-size"
  (for* ([name (in-list '(b50 b10 moons gridclust))]
         [mcs (in-list '(5 12))])
    (test-case (format "~a, min-cluster-size ~a" name mcs)
      (define rows (dataset name))
      (check-equal? (fitted-of rows #:options (hasheq 'min-cluster-size mcs))
                    (fitted-of rows #:options (hasheq 'min-cluster-size mcs 'min-samples mcs))))))

(test-gpu "HD7: identical rows are noise, or one cluster with allow-single"
  (define rows (dataset 'identical))
  (define alone (fitted-of rows))
  (check-equal? (fitted-labels alone) (make-list 100 -1))
  (check-equal? (fitted-persistence alone) '())
  (define single (fitted-of rows #:options (hasheq 'allow-single-cluster #t)))
  (check-equal? (fitted-labels single) (make-list 100 0))
  (check-equal? (fitted-probabilities single) (make-list 100 1.0))
  (test-case "dup x10 lambdas are finite"
    (check-true (andmap (lambda (l) (<= l flt-max)) (fitted-lambdas (fitted-of (dup (dataset 'b50) 10)))))))

(test-gpu "HD8: too few rows give all noise"
  (define rows (dataset 'b50))
  (check-equal? (fitted-labels (fitted-of (take rows 5))) (make-list 5 -1))
  (for ([n (in-list '(2 3))])
    (test-case (format "~a rows" n)
      (check-equal? (fitted-labels (fitted-of (take rows n)
                                              #:options (hasheq 'min-samples 1 'min-cluster-size 2)))
                    (make-list n -1)))))

(test-gpu "HD9: a fit leaves X unchanged"
  (define X (mat (dataset 'moons) 'float32))
  (define before (device-matrix->list* X))
  (hdbscan-fit X)
  (check-equal? (device-matrix->list* X) before))

(test-gpu "HD12: a 1e18 row is noise, and every other label is unchanged"
  (for ([name (in-list '(b50 b10 moons))])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (define labels (fitted-labels (fitted-of (append rows '((1e18 1e18))))))
      (check-equal? labels (append (fitted-labels (fitted-of rows)) '(-1))))))

(test-gpu "P1: one blob is 7 clusters, or 1 with allow-single"
  (define rows (dataset 'one-blob))
  (check-equal? (label-count (fitted-labels (fitted-of rows))) 7)
  (check-equal? (label-count (fitted-labels (fitted-of rows #:options (hasheq 'allow-single-cluster #t))))
                1))

(test-gpu "P2: each ten-fold duplicate of b50 is its own cluster but one"
  (check-equal? (label-count (fitted-labels (fitted-of (dup (dataset 'b50) 10)))) 299))

(test-gpu "P3: b50 doubled is 24 clusters, and a permutation changes the partition"
  (define rows (dup (dataset 'b50) 2))
  (define labels (fitted-labels (fitted-of rows)))
  (check-equal? (label-count labels) 24)
  (define perm (seeded 1 (lambda () (shuffle (range (length rows))))))
  (define permuted (fitted-labels (fitted-of (pick rows perm))))
  (check-pred not (same-partition? labels perm permuted (range (length rows)))))

(test-gpu "P4: epsilon 1e3 on b10 merges to 2 clusters, or 1 with allow-single"
  (define rows (dataset 'b10))
  (check-equal? (label-count (fitted-labels (fitted-of rows
                                                       #:options (hasheq 'cluster-selection-epsilon
                                                                         1e3))))
                2)
  (check-equal? (label-count (fitted-labels
                              (fitted-of rows
                                         #:options (hasheq 'cluster-selection-epsilon
                                                           1e3
                                                           'allow-single-cluster
                                                           #t))))
                1))

(test-gpu "P5: leaf selection on b50 predicts at least 290 of its own fit labels"
  (define X (mat (dataset 'b50) 'float32))
  (define result (hdbscan-fit X #:cluster-selection-method 'leaf))
  (define predicted
    (device-vector->list (hdbscan-prediction-labels (hdbscan-approximate-predict result X))))
  (check-true (>= (count = predicted (device-vector->list (hdbscan-result-labels result))) 290)))
