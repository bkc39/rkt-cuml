#lang racket/base

(require (only-in cuml adjusted-rand-score kmeans-fit kmeans-result-labels)
         (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-fit-transform
                  spectral-embedding-params
                  spectral-embedding-result-embedding
                  spectral-embedding-result-n-neighbors
                  spectral-embedding-result-params
                  spectral-embedding-result?)
         (only-in racket/list append* remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" centred-blobs)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape)
         (only-in "private/spectral-data.rkt"
                  graph->vectors
                  graph-edit
                  graph-n
                  graph-shuffle
                  knn-graph
                  three-blocks)
         (only-in "private/spectral-embedding-data.rkt"
                  column-gaps
                  columns
                  embedded
                  embedded-graph
                  rows-matrix
                  sbm
                  sign-ruled?)
         (only-in "private/umap-data.rkt"
                  blobs
                  far-blobs
                  finite-rows?
                  fixture-labels
                  fixture-rows
                  fixture-X))

(define-values (blocks _truth) (three-blocks))

(define (geometric)
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
      (check-equal?
       (spectral-embedding-result-params result)
       (spectral-embedding-params #:n-components k #:drop-first drop #:norm-laplacian norm)))))

(test-gpu "L1: the default neighbour count is min(256, max(2, ⌊n/10⌋)), and an explicit one wins"
  (for ([entry (in-list (list (list (geometric) #f 2)
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

(define gap-bound 1e-4)

(define constant-bound 4e-5)

(define component-bound 1e-6)

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

(test-gpu "L3, L10: without drop-first, column 0 is constant and the rest is the drop-first fit, bitwise"
  (for ([k (in-list '(1 2 3))])
    (test-case (format "k ~a" k)
      (define kept (on-graph planted (hasheq 'n-components (add1 k) 'drop-first #f)))
      (define dropped (on-graph planted (hasheq 'n-components k)))
      (check-true (sign-ruled? kept))
      (match-define (cons first rest) (columns kept))
      (define spread (/ (- (apply max first) (apply min first)) (apply max (map abs first))))
      (check-true (<= spread constant-bound) (format "spread ~a" spread))
      (check-true (bits-equal? (apply map list rest) dropped)))))

(define (spreads rows truth)
  (for/list ([group (in-list (remove-duplicates truth))])
    (define mine
      (for/list ([row (in-list rows)]
                 [t (in-list truth)]
                 #:when (= t group))
        row))
    (for*/fold ([most 0.0])
               ([a (in-list mine)]
                [b (in-list mine)])
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
  (define labels (kmeans-result-labels (kmeans-fit (rows-matrix rows) #:n-clusters 3 #:n-init 10)))
  (adjusted-rand-score (list->device-vector truth #:dtype 'int32) labels))

(test-gpu "L12: k-means recovers the planted labels from the embedding, for seeds 0 to 9"
  (for ([seed (in-range 10)])
    (check-equal? (ari (embedded (spectral-embedding-fit (rows-matrix (fixture-rows far-blobs))
                                                         #:n-neighbors 10
                                                         #:random-state seed))
                       (fixture-labels far-blobs))
                  1.0
                  (format "far-blobs seed ~a" seed))
    (check-equal? (ari (on-graph planted (hasheq 'random-state seed)) planted-truth)
                  1.0
                  (format "sbm seed ~a" seed))))
