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
         (only-in racket/list last remove-duplicates take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/scaler-data.rkt" bits-equal? matrix-shape scaled-rows transpose)
         (only-in "private/spectral-data.rkt"
                  graph->vectors
                  graph-edit
                  graph-map-vals
                  graph-n
                  graph-shuffle
                  knn-graph
                  three-blocks
                  unit-path
                  unit-rings
                  unit-rings-of)
         (only-in "private/spectral-embedding-data.rkt"
                  column-gaps
                  edge-shares
                  eigen-residuals
                  embedded
                  embedded-graph
                  polygons
                  quantised
                  residual-bound
                  rows-matrix
                  sign-ruled?
                  stochastic-blocks
                  stochastic-graph
                  with-isolated)
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
                             (scaled-rows (fixture-rows far-blobs) (expt 2.0 -70))))])
    (define X (rows-matrix rows))
    (define before (device-matrix->list* X))
    (spectral-embedding-fit X #:n-neighbors 10)
    (check-true (bits-equal? (device-matrix->list* X) before)))
  (define tiny (graph-map-vals (stochastic-graph #:weighted? #t) (lambda (w) (* w (expt 2.0 -20)))))
  (for ([entry (in-list (list (list blocks #t) (list tiny #f)))])
    (match-define (list g norm) entry)
    (define vectors (graph->vectors g))
    (define before (map device-vector->list vectors))
    (apply spectral-embedding-fit-graph (append vectors (list (graph-n g))) #:norm-laplacian norm)
    (check-equal? (map device-vector->list vectors) before (format "norm-laplacian ~a" norm))))

(define gap-bound 1e-4)

(define constant-bound 4e-5)

(define component-bound 1e-6)

(define component-gap 1e-2)

(define-values (stochastic stochastic-truth) (stochastic-blocks '(40 60 80) 0.3 0.01 0))

(define (dense rows n-neighbors)
  (embedded (spectral-embedding-fit (rows-matrix rows) #:n-neighbors n-neighbors)))

(define (on-graph g [options (hasheq)])
  (embedded (embedded-graph g options)))

(test-gpu "L3, L7: 2^k X embeds bitwise as X for k from -120 to 60, sign-ruled"
  (define base (dense quantised 10))
  (check-true (sign-ruled? base))
  (for ([k (in-list '(-120 -70 -45 -30 -1 1 30 60))])
    (check-true (bits-equal? (dense (scaled-rows quantised (expt 2.0 k)) 10) base)
                (format "k ~a" k))))

(test-gpu "L3, L7: 2^k weights embed bitwise as the graph without norm-laplacian, and as 2^(-k/2) times it with"
  (for ([norm (in-list '(#f #t))])
    (define options (hasheq 'norm-laplacian norm))
    (define base (on-graph stochastic options))
    (check-true (sign-ruled? base))
    (for ([k (in-list '(-40 -20 -17 60 100))]
          #:unless (and norm (odd? k)))
      (define scaled (graph-map-vals stochastic (lambda (w) (* w (expt 2.0 k)))))
      (define expected
        (if norm
            (scaled-rows base (expt 2.0 (- (quotient k 2))))
            base))
      (check-true (bits-equal? (on-graph scaled options) expected) (format "norm ~a k ~a" norm k)))))

(test-gpu "L3, L8: a dense fit is the fit of its host kNN graph"
  (define rows (fixture-rows blobs))
  (define ours (dense rows 10))
  (define theirs (on-graph (knn-graph rows 10)))
  (check-true (sign-ruled? ours))
  (check-true (bits-equal? ours theirs) (format "column gaps ~a" (column-gaps ours theirs))))

(test-gpu "L3, L9: shuffling the entries keeps the embedding, and doubled half-weight entries stay near"
  (define base (on-graph stochastic))
  (check-true (sign-ruled? base))
  (for ([seed (in-range 5)])
    (check-true (bits-equal? (on-graph (graph-shuffle stochastic seed)) base)
                (format "seed ~a" seed)))
  (define (halved vals)
    (map (lambda (w) (/ w 2.0)) vals))
  (define doubled
    (graph-edit stochastic
                #:rows (lambda (rows) (append rows rows))
                #:cols (lambda (cols) (append cols cols))
                #:vals (lambda (vals) (append (halved vals) (halved vals)))))
  (check-true (<= (apply max (column-gaps (on-graph doubled) base)) gap-bound)))

(test-gpu "L3, L10: without drop-first, column 0 is constant and the rest is the drop-first fit, bitwise"
  (for ([k (in-list '(1 2 3))])
    (test-case (format "k ~a" k)
      (define kept (on-graph stochastic (hasheq 'n-components (add1 k) 'drop-first #f)))
      (define dropped (on-graph stochastic (hasheq 'n-components k)))
      (check-true (sign-ruled? kept))
      (match-define (cons first rest) (transpose kept))
      (define spread (/ (- (apply max first) (apply min first)) (apply max (map abs first))))
      (check-true (<= spread constant-bound) (format "spread ~a" spread))
      (check-true (bits-equal? (transpose rest) dropped)))))

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

(test-gpu "L11: rings of 30 and 40 embed as eigenvectors, with either Laplacian, at seeds 0 to 4"
  (define g (unit-rings-of '(30 40)))
  (for* ([norm (in-list '(#t #f))]
         [seed (in-range 5)])
    (define rows (embedded (embedded-graph g (hasheq 'norm-laplacian norm 'random-state seed))))
    (define residuals (eigen-residuals g rows #:norm-laplacian norm))
    (check-true (< (apply max residuals) residual-bound)
                (format "norm ~a seed ~a residuals ~a" norm seed residuals))))

(test-gpu "L11: upstream, unchecked: the dense fit of two identical polygons at 3 neighbours is no eigenvector"
  (define rows
    (embedded (spectral-embedding-fit (rows-matrix (polygons 2 30))
                                      #:n-neighbors 3
                                      #:norm-laplacian #f
                                      #:random-state 1)))
  (define residuals (eigen-residuals (unit-rings 2 30) rows #:norm-laplacian #f))
  (check-true (> (apply max residuals) 0.5) (format "residuals ~a" residuals)))

(test-gpu "L11: a path of 12 and 4 samples without edges embeds the path, then one sample alone, for seeds 0 to 9"
  (define g (with-isolated (unit-path 12) 4))
  (for ([seed (in-range 10)])
    (define rows (embedded (embedded-graph g (hasheq 'n-components 6 'random-state seed))))
    (define residuals (eigen-residuals g rows))
    (check-true (< (apply max residuals) residual-bound) (format "seed ~a residuals ~a" seed residuals))
    (define shares (edge-shares g rows))
    (check-true (and (andmap (lambda (s) (> s 0.999)) (take shares 5)) (< (last shares) 1e-9))
                (format "seed ~a shares on the path ~a" seed shares))))

(test-gpu "L11: upstream, unchecked: with 30 samples without edges, no column is on the path of 12"
  (define g (with-isolated (unit-path 12) 30))
  (define shares (edge-shares g (embedded (embedded-graph g (hasheq 'n-components 6)))))
  (check-true (andmap (lambda (s) (< s 1e-9)) shares) (format "shares on the path ~a" shares)))

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
    (check-equal? (ari (on-graph stochastic (hasheq 'random-state seed)) stochastic-truth)
                  1.0
                  (format "sbm seed ~a" seed))))
