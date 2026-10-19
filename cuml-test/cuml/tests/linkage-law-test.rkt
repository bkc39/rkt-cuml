#lang racket/base

(require (only-in racket/list first range remove-duplicates rest second shuffle)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/dbscan-data.rkt" integer-grid mat pick same-partition? seeded)
         (only-in "private/kmeans-data.rkt" blob-data)
         (only-in "private/linkage-data.rkt"
                  children->labels
                  children-cophenetic
                  children-heights
                  connectivities
                  linkage-oracle
                  linked-lists
                  oracle-bounds
                  oracle-cophenetic
                  oracle-partition
                  partition-refines?))

(define (fit-lists rows k connectivity #:c [c 15])
  (linked-lists (mat rows 'float32) #:n-clusters k #:connectivity connectivity #:c c))

(define (children-of rows connectivity #:c [c 15])
  (match-define (list _ children) (fit-lists rows 2 connectivity #:c c))
  children)

(define (labels-of rows k connectivity #:c [c 15])
  (match-define (list labels _) (fit-lists rows k connectivity #:c c))
  labels)

(define (permuted-matrix m p)
  (for/vector ([i (in-list p)])
    (for/vector ([j (in-list p)])
      (vector-ref (vector-ref m i) j))))

(define (non-decreasing? xs)
  (equal? xs (sort xs <)))

(define (grids)
  (list (integer-grid 300 2 20 1) (integer-grid 200 3 8 2) (integer-grid 250 1 400 3)))

(test-gpu "S3: merge heights, recomputed on the host, never decrease down the rows"
  (for* ([connectivity (in-list connectivities)]
         [(rows i) (in-parallel (in-list (grids)) (in-naturals))])
    (test-case (format "~a ~a" connectivity i)
      (check-true (non-decreasing? (map first
                                        (children-heights rows (children-of rows connectivity))))))))

(test-gpu "S4: the labels are the children's cut, roots numbered by descending node"
  (define rows (integer-grid 60 2 12 5))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (define children (children-of rows connectivity))
      (for ([k (in-range 1 61)])
        (check-equal? (labels-of rows k connectivity)
                      (children->labels children 60 k)
                      (format "k ~a" k))))))

(test-gpu "S6: permuting the rows permutes the cophenetic matrix and, at unique cuts, the labels"
  (define rows (integer-grid 200 2 30 6))
  (define p (seeded 6 (lambda () (shuffle (range 200)))))
  (define moved (pick rows p))
  (define o (linkage-oracle rows))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (check-equal? (children-cophenetic moved (children-of moved connectivity))
                    (permuted-matrix (children-cophenetic rows (children-of rows connectivity)) p))
      (for ([k (in-list '(2 3 5 8 13 21))]
            #:when (oracle-partition o k))
        (check-true (same-partition? (labels-of rows k connectivity)
                                     p
                                     (labels-of moved k connectivity)
                                     (range 200))
                    (format "k ~a" k))))))

(test-gpu "S7: integer translations and negation leave the cophenetic matrix unchanged"
  (define rows (integer-grid 200 2 30 7))
  (define (shifted dx dy)
    (for/list ([r (in-list rows)])
      (map + r (list dx dy))))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (define base (children-cophenetic rows (children-of rows connectivity)))
      (for ([moved (in-list (list (shifted 37 -100)
                                  (shifted 100 100)
                                  (shifted -61 5)
                                  (for/list ([r (in-list rows)])
                                    (map - r))))])
        (check-equal? (children-cophenetic moved (children-of moved connectivity)) base)))))

(test-gpu "S8: each cut refines the one above it, and the children do not depend on n-clusters"
  (define rows (integer-grid 40 2 9 8))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (define fits
        (for/list ([k (in-range 1 41)])
          (fit-lists rows k connectivity)))
      (check-equal? (length (remove-duplicates (map second fits))) 1)
      (for ([coarse (in-list fits)]
            [fine (in-list (rest fits))])
        (check-true (partition-refines? (first fine) (first coarse)))))))

(test-gpu "S9: on integer grids, knn with the complete graph has pairwise's cophenetic matrix"
  (for ([rows (in-list (list (integer-grid 200 2 15 9) (integer-grid 150 3 6 10)))])
    (define n (length rows))
    (define o (linkage-oracle rows))
    (test-case (format "~a rows" n)
      (check-equal? (children-cophenetic rows (children-of rows 'knn #:c n))
                    (children-cophenetic rows (children-of rows 'pairwise)))
      (for ([k (in-list '(2 3 4 6 9 14))]
            #:when (oracle-partition o k))
        (check-equal? (labels-of rows k 'knn #:c n)
                      (labels-of rows k 'pairwise)
                      (format "k ~a" k))))))

(test-gpu "S10: every cut lies between the minimum spanning tree's components at its height"
  (define rows (integer-grid 120 2 18 11))
  (define o (linkage-oracle rows))
  (for ([connectivity (in-list connectivities)])
    (test-case (symbol->string connectivity)
      (for ([k (in-range 1 121)])
        (define labels (labels-of rows k connectivity))
        (define-values (finer coarser) (oracle-bounds o k))
        (define all (range 120))
        (check-true (partition-refines? finer labels) (format "k ~a: finer" k))
        (check-true (partition-refines? labels coarser) (format "k ~a: coarser" k))
        (check-equal? (length (remove-duplicates labels)) k)
        (define unique (oracle-partition o k))
        (check-true (or (not unique) (same-partition? labels all unique all))
                    (format "k ~a: unique" k))))))

(test-gpu "S11: the cophenetic matrix is the minimax distance, duplicate rows included"
  (for ([rows (in-list (list (integer-grid 300 2 25 12)
                             (append (integer-grid 100 2 10 13) (integer-grid 40 2 10 13))
                             (integer-grid 200 4 4 14)))])
    (define expected (oracle-cophenetic (linkage-oracle rows)))
    (for ([connectivity (in-list connectivities)])
      (test-case (format "~a rows ~a" (length rows) connectivity)
        (check-equal? (children-cophenetic rows (children-of rows connectivity)) expected)))))

(test-gpu "P3: far from the origin, float32's expanded distances lose the cophenetic matrix"
  (define rows (integer-grid 200 2 20 6))
  (define far
    (for/list ([r (in-list rows)])
      (map (lambda (x) (+ x 10000)) r)))
  (define expected (oracle-cophenetic (linkage-oracle far)))
  (define found (children-cophenetic far (children-of far 'pairwise)))
  (define worst
    (for*/fold ([worst 0])
               ([(row i) (in-parallel (in-vector found) (in-naturals))]
                [j (in-range 200)])
      (max worst (abs (- (sqrt (vector-ref row j)) (sqrt (vector-ref (vector-ref expected i) j)))))))
  (check-true (> worst 1) (format "the largest cophenetic error is ~a" worst)))

(test-gpu "P4: on float data, knn with the complete graph can cut apart from pairwise"
  (define X (blob-data 500 3 1 0 'float32 #:std 1.0))
  (define unique (oracle-partition (linkage-oracle (device-matrix->list* X)) 393))
  (define all (range 500))
  (define (cut connectivity)
    (first (linked-lists X #:n-clusters 393 #:connectivity connectivity #:c 500)))
  (check-true (same-partition? (cut 'pairwise) all unique all))
  (check-false (same-partition? (cut 'knn) all unique all)))

(test-gpu "P5: a c below -7 fits 200 rows as -7, and 2 neighbours are not the exact tree"
  (define X (blob-data 200 2 1 0 'float32 #:std 1.0))
  (define least (linked-lists X #:connectivity 'knn #:c -7))
  (for ([c (in-list '(-8 -31 -2147483648))])
    (check-equal? (linked-lists X #:connectivity 'knn #:c c) least (format "c ~a" c)))
  (define rows (integer-grid 200 2 15 9))
  (check-not-equal? (children-cophenetic rows (children-of rows 'knn #:c -7))
                    (oracle-cophenetic (linkage-oracle rows))))
