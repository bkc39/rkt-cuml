#lang racket/base

(require (only-in cuml/cluster single-linkage-fit single-linkage-result)
         (only-in racket/list make-list range)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list* dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" integer-grid mat same-partition?)
         (only-in "private/kmeans-data.rkt" blob-data)
         (only-in "private/linkage-data.rkt"
                  group-rows
                  hand-children
                  hand-labels
                  hand-rows
                  linked-lists
                  permuted-children
                  permuted-labels
                  permuted-rows
                  result->lists
                  tie-children
                  tie-labels
                  tie-rows
                  valid-dendrogram-problems))

(define connectivities '(pairwise knn))

(test-gpu "S1: the hand grid's children and labels at every k, as given and permuted"
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (for ([k (in-range 1 6)]
            [labels (in-list hand-labels)]
            [permuted (in-list permuted-labels)])
        (match-define (single-linkage-result found children)
          (single-linkage-fit (mat hand-rows 'float32) #:n-clusters k #:connectivity connectivity))
        (check-equal? (result->lists (single-linkage-result found children))
                      (list labels hand-children))
        (check-equal? (list (shape found) (dtype found) (shape children) (dtype children))
                      '((5) int32 (4 2) int32))
        (check-equal?
         (linked-lists (mat permuted-rows 'float32) #:n-clusters k #:connectivity connectivity)
         (list permuted permuted-children))))))

(define (structure-data)
  (append (for/list ([seed (in-range 3)])
            (mat (integer-grid 300 2 25 seed) 'float32))
          (list (blob-data 500 3 4 7 'float32) (blob-data 400 5 6 8 'float32 #:std 0.3))))

(test-gpu "S2: the children form a dendrogram, each row ascending"
  (for* ([connectivity (in-list connectivities)]
         [(X i) (in-parallel (in-list (structure-data)) (in-naturals))])
    (test-case (format "~a ~a" connectivity i)
      (match-define (list _ children) (linked-lists X #:connectivity connectivity))
      (check-equal? (valid-dendrogram-problems children (length (device-matrix->list* X))) '()))))

(test-gpu "S5: one cluster, one cluster per sample, one sample and two samples"
  (define X (mat (integer-grid 50 2 10 4) 'float32))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (define (labels-at k)
        (match-define (list labels _) (linked-lists X #:n-clusters k #:connectivity connectivity))
        labels)
      (check-equal? (labels-at 1) (make-list 50 0))
      (check-equal? (labels-at 50) (reverse (range 50)))
      (match-define (single-linkage-result one one-children)
        (single-linkage-fit (mat '((3 4)) 'float32) #:n-clusters 1 #:connectivity connectivity))
      (check-equal? (result->lists (single-linkage-result one one-children)) '((0) ()))
      (check-equal? (shape one-children) '(0 2))
      (define two (mat '((1 2) (5 3)) 'float32))
      (check-equal? (linked-lists two #:n-clusters 1 #:connectivity connectivity) '((0 0) ((0 1))))
      (check-equal? (linked-lists two #:n-clusters 2 #:connectivity connectivity) '((1 0) ((0 1)))))))

(test-gpu "P1: ties come in cuML's order, the same under both connectivities"
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (check-equal? (linked-lists (mat tie-rows 'float32) #:n-clusters 3 #:connectivity connectivity)
                    (list tie-labels tie-children)))))

(test-gpu "P2: knn cannot join 1024 separated groups at c 15, and joins them at c 20"
  (define rows (group-rows 1024 32))
  (define X (mat rows 'float32))
  (check-cuml-error 'logic
                    #rx"^single-linkage-fit: .*KNN graph could not be connected in 10 iterations"
                    (lambda () (single-linkage-fit X)))
  (match-define (list labels _) (linked-lists X #:c 20))
  (define top
    (for/list ([r (in-list rows)])
      (if (> (list-ref r 9) 5) 1 0)))
  (define all (range (length rows)))
  (check-equal? (same-partition? labels all top all) #t))
