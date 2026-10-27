#lang racket/base

(require (only-in cuml adjusted-rand-score kmeans-fit kmeans-result-labels)
         (only-in cuml/manifold spectral-embedding-fit)
         (only-in racket/list append* remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" centred-blobs)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/spectral-data.rkt" graph-edit graph-shuffle knn-graph three-blobs)
         (only-in "private/spectral-embedding-data.rkt"
                  column-gaps
                  columns
                  embedded
                  embedded-graph
                  rows-matrix
                  sbm
                  sign-ruled?)
         (only-in "private/umap-data.rkt" blobs far-blobs fixture-labels fixture-rows))

(define gap-bound 1e-4)

(define constant-bound 1e-5)

(define component-bound 1e-3)

(define component-gap 1e-2)

(define-values (planted planted-truth) (sbm '(40 60 80) 0.3 0.01 0))

(define (dense rows n-neighbors)
  (embedded (spectral-embedding-fit (rows-matrix rows) #:n-neighbors n-neighbors)))

(define (on-graph g [options (hasheq)])
  (embedded (embedded-graph g options)))

(define quantised
  (let-values ([(rows _labels) (centred-blobs '((0 0 0) (1 1 0) (0 1 1)) 40 7)])
    (define most (apply max (map abs (append* rows))))
    (for/list ([row (in-list rows)])
      (for/list ([x (in-list row)])
        (/ (round (* (/ x most) 1.5 (expt 2 20))) (expt 2.0 20))))))

(test-gpu "L3, L7: 2^k X embeds bitwise as X for k from -120 to 60, sign-ruled"
  (define base (dense quantised 10))
  (check-true (sign-ruled? base))
  (for ([k (in-list '(-120 -70 -45 -30 -1 1 30 60))])
    (define scaled
      (for/list ([row (in-list quantised)])
        (for/list ([x (in-list row)])
          (* x (expt 2.0 k)))))
    (check-true (bits-equal? (dense scaled 10) base) (format "k ~a" k))))

(test-gpu "L3, L8: a dense fit is the fit of its host kNN graph"
  (define rows (fixture-rows blobs))
  (define ours (dense rows 10))
  (define theirs (on-graph (knn-graph rows 10)))
  (check-true (sign-ruled? ours))
  (check-true (bits-equal? ours theirs) (format "column gaps ~a" (column-gaps ours theirs))))

(test-gpu "L3, L9: shuffling the entries keeps the embedding, and doubled half-weight entries stay near"
  (define base (on-graph planted))
  (check-true (sign-ruled? base))
  (for ([seed (in-range 5)])
    (check-true (bits-equal? (on-graph (graph-shuffle planted seed)) base) (format "seed ~a" seed)))
  (define (halved vals)
    (map (lambda (w) (/ w 2.0)) vals))
  (define doubled
    (graph-edit planted
                #:rows (lambda (rows) (append rows rows))
                #:cols (lambda (cols) (append cols cols))
                #:vals (lambda (vals) (append (halved vals) (halved vals)))))
  (check-true (<= (apply max (column-gaps (on-graph doubled) base)) gap-bound)))

(test-gpu "L3, L10: without drop-first, column 0 is constant and the rest is the drop-first fit"
  (for ([k (in-list '(1 2 3))])
    (test-case (format "k ~a" k)
      (define kept (on-graph planted (hasheq 'n-components (add1 k) 'drop-first #f)))
      (define dropped (on-graph planted (hasheq 'n-components k)))
      (check-true (sign-ruled? kept))
      (match-define (cons first rest) (columns kept))
      (define spread (/ (- (apply max first) (apply min first)) (apply max (map abs first))))
      (check-true (<= spread constant-bound) (format "spread ~a" spread))
      (check-true (<= (apply max (column-gaps (apply map list rest) dropped)) gap-bound)))))

(define (spreads rows truth)
  (for/list ([group (in-list (remove-duplicates truth))])
    (define mine
      (for/list ([row (in-list rows)]
                 [t (in-list truth)]
                 #:when (= t group))
        row))
    (for*/fold ([most 0.0]) ([a (in-list mine)] [b (in-list mine)])
      (max most (apply max (map (lambda (x y) (abs (- x y))) a b))))))

(test-gpu "L3, L11: on far-blobs' three components, each component's rows agree"
  (define rows (dense (fixture-rows far-blobs) 10))
  (check-true (sign-ruled? rows))
  (define within (spreads rows (fixture-labels far-blobs)))
  (check-true (<= (apply max within) component-bound) (format "spreads ~a" within))
  (define firsts
    (for/list ([group (in-list '(0 1 2))])
      (for/first ([row (in-list rows)]
                  [t (in-list (fixture-labels far-blobs))]
                  #:when (= t group))
        row)))
  (for* ([a (in-list firsts)]
         [b (in-list firsts)]
         #:unless (eq? a b))
    (check-true (> (apply max (map (lambda (x y) (abs (- x y))) a b)) component-gap))))

(define (ari rows truth)
  (define labels
    (kmeans-result-labels (kmeans-fit (rows-matrix rows) #:n-clusters 3 #:n-init 10)))
  (adjusted-rand-score (list->device-vector truth #:dtype 'int32) labels))

(test-gpu "L12: k-means recovers the planted labels from the embedding, for seeds 0 to 9"
  (define-values (three truth) (three-blobs))
  (for ([seed (in-range 10)])
    (check-equal? (ari (embedded (spectral-embedding-fit (rows-matrix three) #:random-state seed))
                       truth)
                  1.0
                  (format "three-blobs seed ~a" seed))
    (check-equal? (ari (on-graph planted (hasheq 'random-state seed)) planted-truth)
                  1.0
                  (format "sbm seed ~a" seed))))
