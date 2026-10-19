#lang racket/base

(require (only-in cuml/cluster
                  single-linkage-fit
                  single-linkage-result-children
                  single-linkage-result-labels)
         (only-in racket/list argmin first remove-duplicates)
         (only-in racket/match == match match-define)
         (only-in racket/math sqr)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "dbscan-data.rkt" seeded))

(provide (struct-out oracle)
         children->labels
         children-cophenetic
         children-heights
         group-rows
         hand-children
         hand-labels
         hand-rows
         linkage-oracle
         linked-lists
         oracle-bounds
         oracle-partition
         partition-refines?
         permuted-children
         permuted-labels
         permuted-rows
         result->lists
         squared-distance
         tie-children
         tie-labels
         tie-rows
         two-far
         valid-dendrogram-problems)

(define hand-rows '((0) (1) (3) (7) (15)))
(define hand-children '((0 1) (2 5) (3 6) (4 7)))
(define hand-labels '((0 0 0 0 0) (0 0 0 0 1) (0 0 0 2 1) (0 0 3 2 1) (4 3 2 1 0)))
(define permuted-rows '((15) (3) (0) (7) (1)))
(define permuted-children '((2 4) (1 5) (3 6) (0 7)))
(define permuted-labels '((0 0 0 0 0) (1 0 0 0 0) (2 0 0 1 0) (3 2 0 1 0) (4 3 2 1 0)))

(define tie-rows
  (for/list ([i (in-range 8)])
    (list i)))
(define tie-children '((0 1) (2 8) (3 9) (4 5) (6 11) (7 12) (10 13)))
(define tie-labels '(1 1 1 1 0 0 0 2))

(define (result->lists r)
  (list (device-vector->list (single-linkage-result-labels r))
        (device-matrix->list* (single-linkage-result-children r))))

(define linked-lists (compose1 result->lists single-linkage-fit))

(define (group-rows groups per)
  (for*/list ([g (in-range groups)]
              [m (in-range per)])
    (for/list ([i (in-range 10)])
      (+ (* (bitwise-and (arithmetic-shift g (- i)) 1) (add1 i))
         (if (= i 0)
             (* 1e-3 m)
             0)))))

(define (two-far seed)
  (seeded seed
          (lambda ()
            (for*/list ([offset (in-list '(0 100))]
                        [_ (in-range 40)])
              (for/list ([_ (in-range 2)])
                (+ offset (/ (random 100000) 9973.0)))))))

(define (squared-distance a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (sqr (- x y))))

(struct oracle (n weights edges cophenetic))

(define (prim-edges points)
  (define n (vector-length points))
  (define best (make-vector n +inf.0))
  (define from (make-vector n #f))
  (define in-tree (make-vector n #f))
  (vector-set! best 0 0)
  (for/fold ([edges '()]
             #:result (reverse edges))
            ([_ (in-range n)])
    (define next
      (argmin (lambda (i) (vector-ref best i))
              (for/list ([i (in-range n)]
                         #:unless (vector-ref in-tree i))
                i)))
    (vector-set! in-tree next #t)
    (for ([j (in-range n)]
          #:unless (vector-ref in-tree j))
      (define d (squared-distance (vector-ref points next) (vector-ref points j)))
      (define closer? (< d (vector-ref best j)))
      (vector-set! best
                   j
                   (if closer?
                       d
                       (vector-ref best j)))
      (vector-set! from
                   j
                   (if closer?
                       next
                       (vector-ref from j))))
    (match (vector-ref from next)
      [#f edges]
      [parent (cons (list (vector-ref best next) parent next) edges)])))

(define (union-find n)
  (define parent (build-vector n values))
  (define (find i)
    (match (vector-ref parent i)
      [(== i) i]
      [p
       (define root (find p))
       (vector-set! parent i root)
       root]))
  (values find (lambda (a b) (vector-set! parent (find a) (find b)))))

(define (merge-heights n merges)
  (define cophenetic
    (for/vector #:length n
                ([_ (in-range n)])
      (make-vector n 0)))
  (define members (build-vector n list))
  (define-values (find join!) (union-find n))
  (for ([m (in-list merges)])
    (match-define (list height (app find ra) (app find rb)) m)
    (for* ([i (in-list (vector-ref members ra))]
           [j (in-list (vector-ref members rb))])
      (vector-set! (vector-ref cophenetic i) j height)
      (vector-set! (vector-ref cophenetic j) i height))
    (join! ra rb)
    (vector-set! members (find ra) (append (vector-ref members ra) (vector-ref members rb))))
  cophenetic)

(define (linkage-oracle rows)
  (define points (list->vector rows))
  (define edges (sort (prim-edges points) < #:key first))
  (oracle (vector-length points)
          (sort (map first edges) >)
          edges
          (merge-heights (length rows) edges)))

(define (kept-components o keep?)
  (define-values (find join!) (union-find (oracle-n o)))
  (for ([e (in-list (oracle-edges o))]
        #:when (keep? (first e)))
    (match-define (list _ a b) e)
    (join! a b))
  (map find (build-list (oracle-n o) values)))

(define (oracle-bounds o k)
  (match k
    [1
     (define all (kept-components o (lambda (_) #t)))
     (values all all)]
    [_
     (define cut (list-ref (oracle-weights o) (- k 2)))
     (values (kept-components o (lambda (w) (< w cut)))
             (kept-components o (lambda (w) (<= w cut))))]))

(define (oracle-partition o k)
  (define weights (oracle-weights o))
  (define-values (finer _) (oracle-bounds o k))
  (and (or (= k 1) (= k (oracle-n o)) (> (list-ref weights (- k 2)) (list-ref weights (- k 1))))
       finer))

(define (partition-refines? fine coarse)
  (define seen (make-hash))
  (for/and ([f (in-list fine)]
            [c (in-list coarse)])
    (equal? (hash-ref! seen f c) c)))

(define (children->labels children n k)
  (define-values (find join!) (union-find (+ n n)))
  (for ([pair (in-list children)]
        [i (in-range (- n k))])
    (match-define (list a b) pair)
    (join! a (+ n i))
    (join! b (+ n i)))
  (define roots (remove-duplicates (map find (build-list n values))))
  (define label
    (for/hash ([root (in-list (sort roots >))]
               [l (in-naturals)])
      (values root l)))
  (for/list ([i (in-range n)])
    (hash-ref label (find i))))

(define (children-heights rows children)
  (define points (list->vector rows))
  (define n (vector-length points))
  (define members (make-vector (+ n n) '()))
  (for ([i (in-range n)])
    (vector-set! members i (list i)))
  (for/list ([pair (in-list children)]
             [i (in-naturals)])
    (match-define (list (app (lambda (id) (vector-ref members id)) a)
                        (app (lambda (id) (vector-ref members id)) b))
      pair)
    (vector-set! members (+ n i) (append a b))
    (list (apply min
                 (for*/list ([x (in-list a)]
                             [y (in-list b)])
                   (squared-distance (vector-ref points x) (vector-ref points y))))
          (first a)
          (first b))))

(define (children-cophenetic rows children)
  (merge-heights (length rows) (children-heights rows children)))

(define (valid-dendrogram-problems children n)
  (define ids (apply append children))
  (append (if (andmap (lambda (pair) (apply < pair)) children)
              '()
              '("each row ascending with distinct entries"))
          (if (for/and ([pair (in-list children)]
                        [i (in-naturals)])
                (< (apply max pair) (+ n i)))
              '()
              '("row i's entries below n+i"))
          (if (equal? (sort ids <) (build-list (max 0 (- (+ n n) 2)) values))
              '()
              '("every id 0..2n-3 exactly once"))))
