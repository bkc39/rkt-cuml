#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/cluster
                  hdbscan-condensed-tree-children
                  hdbscan-condensed-tree-lambdas
                  hdbscan-condensed-tree-parents
                  hdbscan-condensed-tree-selected-clusters
                  hdbscan-condensed-tree-sizes
                  hdbscan-fit
                  hdbscan-linkage-tree-children
                  hdbscan-linkage-tree-distances
                  hdbscan-linkage-tree-sizes
                  hdbscan-prediction-labels
                  hdbscan-prediction-probabilities
                  hdbscan-result-cluster-persistence
                  hdbscan-result-condensed-tree
                  hdbscan-result-labels
                  hdbscan-result-minimum-spanning-tree
                  hdbscan-result-probabilities
                  hdbscan-result-single-linkage-tree
                  hdbscan-spanning-tree-destinations
                  hdbscan-spanning-tree-sources
                  hdbscan-spanning-tree-weights)
         (only-in racket/contract/base -> contract-out listof or/c)
         (only-in racket/file file->value)
         (only-in racket/list append* last list-set make-list range remove-duplicates)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/math pi)
         (only-in racket/runtime-path define-runtime-path)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "dbscan-data.rkt" mat seeded)
         (only-in "estimator-data.rkt" apply-options)
         (only-in "spectral-data.rkt" blobs circles float32-rows moons))

(provide (struct-out fitted)
         (contract-out [dataset (-> (apply or/c dataset-names) (listof (listof real?)))])
         alternate
         alternate-matrix
         close?
         dataset-names
         descent-problems
         dup
         fit-lists
         fitted->list
         fitted-of
         label-count
         prediction->lists
         same-fit?
         structure-problems
         without-persistence)

(struct fitted
        (labels probabilities
                persistence
                parents
                children
                lambdas
                sizes
                selected
                linkage
                distances
                linkage-sizes
                sources
                destinations
                weights)
  #:transparent)

(define-runtime-path normals-path "normals-0.rktd")

(define normals (file->value normals-path))

(define (numpy-blobs gap)
  (float32-rows (for/list ([row (in-list normals)]
                           [i (in-naturals)])
                  (define shift (* gap (quotient i 100)))
                  (map (lambda (x) (+ x shift)) row))))

(define dataset-names
  '(b50 b10 b5d4 varied overlap moons circles gridclust grid30 one-blob identical))

(define (gaussian)
  (* (sqrt (* -2.0 (log (- 1.0 (random))))) (cos (* 2.0 pi (random)))))

(define (varied-blobs n-each sds gap seed)
  (seeded seed
          (lambda ()
            (float32-rows (for*/list ([(sd g) (in-indexed sds)]
                                      [_ (in-range n-each)])
                            (for/list ([_ (in-range 2)])
                              (+ (* g gap) (* sd (gaussian)))))))))

(define (integer-grid side [offset 0])
  (for*/list ([i (in-range side)]
              [j (in-range side)])
    (list (+ offset i) (+ offset j))))

(define (first-value thunk)
  (call-with-values thunk (lambda (rows . _) rows)))

(define (dataset name)
  (case name
    [(b50) (numpy-blobs 50.0)]
    [(b10) (numpy-blobs 10.0)]
    [(b5d4) (first-value (lambda () (blobs 200 4 4 5.0 3)))]
    [(varied) (varied-blobs 500 '(1.0 2.5 0.5) 10.0 170)]
    [(overlap) (first-value (lambda () (blobs 2000 5 8 3.0 1)))]
    [(moons) (first-value (lambda () (moons 1000 0.05 0)))]
    [(circles) (first-value (lambda () (circles 1000 0.5 0.03 0)))]
    [(gridclust)
     (float32-rows (append* (for/list ([k (in-range 3)])
                              (integer-grid 10 (* 100 k)))))]
    [(grid30) (float32-rows (integer-grid 30))]
    [(one-blob) (numpy-blobs 0.0)]
    [(identical) (make-list 100 '(1.0 1.0))]))

(define (dup rows times)
  (for*/list ([row (in-list rows)]
              [_ (in-range times)])
    row))

(define (alternate rows parity)
  (for/list ([row (in-list rows)]
             [i (in-naturals)]
             #:when (= (modulo i 2) parity))
    row))

(define (alternate-matrix name parity)
  (mat (alternate (dataset name) parity) 'float32))

(define (label-count labels)
  (add1 (apply max -1 labels)))

(define (fit-lists result)
  (define tree (hdbscan-result-condensed-tree result))
  (define linkage (hdbscan-result-single-linkage-tree result))
  (define mst (hdbscan-result-minimum-spanning-tree result))
  (fitted (device-vector->list (hdbscan-result-labels result))
          (device-vector->list (hdbscan-result-probabilities result))
          (device-vector->list (hdbscan-result-cluster-persistence result))
          (device-vector->list (hdbscan-condensed-tree-parents tree))
          (device-vector->list (hdbscan-condensed-tree-children tree))
          (device-vector->list (hdbscan-condensed-tree-lambdas tree))
          (device-vector->list (hdbscan-condensed-tree-sizes tree))
          (device-vector->list (hdbscan-condensed-tree-selected-clusters tree))
          (device-matrix->list* (hdbscan-linkage-tree-children linkage))
          (device-vector->list (hdbscan-linkage-tree-distances linkage))
          (device-vector->list (hdbscan-linkage-tree-sizes linkage))
          (device-vector->list (hdbscan-spanning-tree-sources mst))
          (device-vector->list (hdbscan-spanning-tree-destinations mst))
          (device-vector->list (hdbscan-spanning-tree-weights mst))))

(define (fitted-of rows #:options [options (hasheq)])
  (fit-lists (apply-options hdbscan-fit options (mat rows 'float32))))

(define (prediction->lists p)
  (list (device-vector->list (hdbscan-prediction-labels p))
        (device-vector->list (hdbscan-prediction-probabilities p))))

(define (problem ok? text)
  (and (not ok?) text))

(define (structure-problems f n)
  (match-define (fitted labels
                        probabilities
                        persistence
                        parents
                        children
                        _lambdas
                        sizes
                        selected
                        linkage
                        distances
                        linkage-sizes
                        sources
                        destinations
                        weights)
    f)
  (define k (length selected))
  (define points
    (for/list ([c (in-list children)]
               #:when (< c n))
      c))
  (filter
   values
   (list (problem (= (length labels) n) "labels: one per row")
         (problem (andmap (lambda (l) (<= -1 l (sub1 k))) labels) "labels: in [-1, k)")
         (problem (= (length (remove-duplicates (filter (lambda (l) (>= l 0)) labels))) k)
                  "labels: every cluster occurs")
         (problem (andmap (lambda (p) (<= 0.0 p 1.0)) probabilities) "probabilities: in [0, 1]")
         (problem (andmap (lambda (l p)
                            (if (= l -1)
                                (= p 0.0)
                                (> p 0.0)))
                          labels
                          probabilities)
                  "probabilities: 0 exactly on noise, positive elsewhere")
         (problem (= (length persistence) k) "persistence: one per cluster")
         (problem (andmap (lambda (p) (>= p n)) parents) "condensed: parents are clusters")
         (problem (= (apply max (append parents children)) (length parents))
                  "condensed: the largest id is the edge count")
         (problem (equal? (sort points <) (range n)) "condensed: each point is a child once")
         (problem (for/and ([c (in-list children)]
                            [s (in-list sizes)]
                            #:when (< c n))
                    (= s 1))
                  "condensed: leaf edges have size 1")
         (problem (= (length linkage) (sub1 n)) "linkage: n-1 merges")
         (problem (andmap (match-lambda
                            [(list a b) (<= 0 a b (- (* 2 n) 2))])
                          linkage)
                  "linkage: sorted pairs of ids in [0, 2n-2]")
         (problem (sorted? distances) "linkage: distances do not decrease")
         (problem (= (last linkage-sizes) n) "linkage: the last merge holds every row")
         (problem (= (length sources) (length destinations) (sub1 n)) "spanning tree: n-1 edges")
         (problem (equal? weights distances) "spanning tree: weights are the linkage distances"))))

(define (sorted? xs)
  (or (null? xs) (apply <= xs)))

(define (descent-problems f n)
  (define parent-of
    (for/hasheqv ([c (in-list (fitted-children f))]
                  [p (in-list (fitted-parents f))])
      (values c p)))
  (define (ancestors i)
    (match (hash-ref parent-of i #f)
      [#f '()]
      [p (cons p (ancestors p))]))
  (define selected
    (for/vector ([s (in-list (fitted-selected f))])
      (+ n s)))
  (filter values
          (cons (problem (sorted-strictly? (vector->list selected))
                         "selected clusters: strictly ascending")
                (for/list ([l (in-list (fitted-labels f))]
                           [i (in-naturals)])
                  (define above
                    (for/list ([s (in-vector selected)]
                               #:when (memv s (ancestors i)))
                      s))
                  (problem (equal? above
                                   (if (= l -1)
                                       '()
                                       (list (vector-ref selected l))))
                           (format "point ~a: label ~a, under selected ~a" i l above))))))

(define (sorted-strictly? xs)
  (or (null? xs) (apply < xs)))

(define (close? a b tolerance)
  (and (= (length a) (length b))
       (for/and ([x (in-list a)]
                 [y (in-list b)])
         (<= (abs (- x y)) (* tolerance (max (abs x) (abs y) 1e-30))))))

(define fitted-fields
  (list fitted-labels
        fitted-probabilities
        fitted-persistence
        fitted-parents
        fitted-children
        fitted-lambdas
        fitted-sizes
        fitted-selected
        fitted-linkage
        fitted-distances
        fitted-linkage-sizes
        fitted-sources
        fitted-destinations
        fitted-weights))

(define (fitted->list f)
  (map (lambda (field) (field f)) fitted-fields))

(define (without-persistence fields)
  (list-set fields 2 '()))

(define (same-fit? a b)
  (and (equal? (without-persistence (fitted->list a)) (without-persistence (fitted->list b)))
       (close? (fitted-persistence a) (fitted-persistence b) 1e-5)))
