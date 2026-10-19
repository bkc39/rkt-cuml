#lang racket/base

(require (only-in cuml/cluster dbscan-fit dbscan-result-core-sample-indices dbscan-result-labels)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in racket/math sqr)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         (only-in "fitted.rkt" fitted))

(provide (struct-out expected)
         canonical
         cosine-cloud
         cosine-within
         dbscan-oracle
         distance-matrix
         euclidean-within
         first-copies
         fitted-lists
         grid-labels
         grid-core
         grid-rows
         grid-weights
         grid-weighted-core
         integer-grid
         labels-at
         mat
         repeat-rows
         result->lists
         same-partition?
         seeded
         structure-problems
         vec)

(struct expected (labels core options))

(define grid-rows '((0) (1) (2) (4) (6) (7) (8) (20)))
(define grid-labels '(0 0 0 0 1 1 1 -1))
(define grid-core '(2 4))
(define grid-weights '(4 1 1 1 1 1 1 1))
(define grid-weighted-core '(0 1 2 4))

(define (mat rows dt)
  (list*->device-matrix rows #:dtype dt))

(define (vec xs dt)
  (list->device-vector xs #:dtype dt))

(define (result->lists r)
  (list (device-vector->list (dbscan-result-labels r))
        (device-vector->list (dbscan-result-core-sample-indices r))))

(define fitted-lists (fitted dbscan-fit (lambda (run) (result->lists (run)))))

(define (labels-at labels indices)
  (define at (list->vector labels))
  (for/list ([i (in-list indices)])
    (vector-ref at i)))

(define (seeded seed thunk)
  (parameterize ([current-pseudo-random-generator
                  (vector->pseudo-random-generator (vector seed 7 11 13 17 19))])
    (thunk)))

(define (integer-grid n d extent seed)
  (seeded seed
          (lambda ()
            (for/list ([_ (in-range n)])
              (for/list ([_ (in-range d)])
                (random extent))))))

(define (squared-distance a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (sqr (- x y))))

(define ((euclidean-within rows eps) i j)
  (<= (squared-distance (vector-ref rows i) (vector-ref rows j)) (sqr eps)))

(define (norm v)
  (sqrt (for/sum ([x (in-list v)]) (sqr (exact->inexact x)))))

(define (cosine-distance a b)
  (- 1.0
     (/ (for/sum ([x (in-list a)] [y (in-list b)]) (* (exact->inexact x) y)) (* (norm a) (norm b)))))

(define ((cosine-within rows eps) i j)
  (<= (cosine-distance (vector-ref rows i) (vector-ref rows j)) eps))

(define (distance-matrix rows)
  (for/list ([a (in-list rows)])
    (for/list ([b (in-list rows)])
      (sqrt (exact->inexact (squared-distance a b))))))

(define (dbscan-oracle n within? min-samples #:weights [weights #f])
  (define weight
    (if weights
        (list->vector weights)
        (make-vector n 1)))
  (define neighbours
    (for/vector #:length n
                ([i (in-range n)])
      (for/list ([j (in-range n)]
                 #:when (within? i j))
        j)))
  (define core?
    (for/vector #:length n
                ([ns (in-vector neighbours)])
      (>= (for/sum ([j (in-list ns)]) (vector-ref weight j)) min-samples)))
  (define labels (make-vector n -1))
  (for/fold ([next 0])
            ([i (in-range n)]
             #:when (and (vector-ref core? i) (= (vector-ref labels i) -1)))
    (let spread ([frontier (list i)])
      (cond
        [(null? frontier) (void)]
        [else
         (for ([j (in-list frontier)])
           (vector-set! labels j next))
         (spread (remove-duplicates (for*/list ([j (in-list frontier)]
                                                [m (in-list (vector-ref neighbours j))]
                                                #:when (vector-ref core? m)
                                                #:when (= (vector-ref labels m) -1))
                                      m)))]))
    (add1 next))
  (define options
    (for/vector #:length n
                ([i (in-range n)])
      (cond
        [(vector-ref core? i) '()]
        [else
         (sort (remove-duplicates (for/list ([j (in-list (vector-ref neighbours i))]
                                             #:when (vector-ref core? j))
                                    (vector-ref labels j)))
               <)])))
  (for ([i (in-naturals)]
        [choices (in-vector options)])
    (match choices
      [(cons smallest _) (vector-set! labels i smallest)]
      ['() (void)]))
  (expected (vector->list labels)
            (for/list ([i (in-range n)]
                       #:when (vector-ref core? i))
              i)
            options))

(define (canonical labels indices)
  (define picked (labels-at labels indices))
  (define names
    (for/hash ([l (in-list (remove-duplicates picked))]
               [k (in-naturals)])
      (values l k)))
  (map (lambda (l) (hash-ref names l)) picked))

(define (same-partition? a a-indices b b-indices)
  (equal? (canonical a a-indices) (canonical b b-indices)))

(define (repeat-rows rows weights)
  (for*/list ([(row w) (in-parallel (in-list rows) (in-list weights))]
              [_ (in-range w)])
    row))

(define (first-copies weights)
  (for/fold ([at 0]
             [firsts '()]
             #:result (reverse firsts))
            ([w (in-list weights)])
    (values (+ at w) (cons at firsts))))

(define (structure-problems labels core)
  (define k (add1 (apply max -1 labels)))
  (define core-labels (labels-at labels core))
  (define first-core
    (for/list ([c (in-range k)])
      (for/first ([i (in-list core)]
                  [l (in-list core-labels)]
                  #:when (= l c))
        i)))
  (append (if (equal? core (sort (remove-duplicates core) <))
              '()
              '("core indices strictly ascending"))
          (if (andmap (lambda (l) (>= l 0)) core-labels)
              '()
              '("core points labelled"))
          (if (andmap (lambda (l) (<= -1 l (sub1 k))) labels)
              '()
              '("labels in {-1} and [0, k)"))
          (if (for/and ([c (in-range k)])
                (memv c labels))
              '()
              '("every label in [0, k) used"))
          (if (and (andmap values first-core) (equal? first-core (sort first-core <)))
              '()
              '("clusters numbered by smallest core index"))))

(define (cosine-cloud n d eps seed #:margin [margin 1e-3])
  (seeded seed
          (lambda ()
            (define centres
              (for/list ([_ (in-range 4)])
                (for/list ([_ (in-range d)])
                  (- (random 33) 16))))
            (define (candidate)
              (define centre (list-ref centres (random 4)))
              (for/list ([c (in-list centre)])
                (+ c (/ (- (random 129) 64) 32))))
            (let grow ([rows '()]
                       [count 0])
              (cond
                [(= count n) (reverse rows)]
                [else
                 (define row (candidate))
                 (if (and (positive? (norm row))
                          (for/and ([other (in-list rows)])
                            (> (abs (- (cosine-distance row other) eps)) margin)))
                     (grow (cons row rows) (add1 count))
                     (grow rows count))])))))
