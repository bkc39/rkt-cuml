#lang racket/base

(require (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-result-embedding
                  spectral-embedding-result-n-neighbors
                  spectral-embedding-result-params)
         (only-in racket/flonum flsingle)
         (only-in racket/list append* argmax make-list second)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/math pi sqr)
         racket/runtime-path
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list* list*->device-matrix)
         (only-in "blob-data.rkt" centred-blobs)
         (only-in "dbscan-data.rkt" seeded)
         (only-in "estimator-data.rkt" apply-options)
         (only-in "python-env.rkt" run-cases)
         (only-in "scaler-data.rkt" bits-equal? dot transpose)
         (only-in "spectral-data.rkt"
                  graph
                  graph->vectors
                  graph-cols
                  graph-edit
                  graph-map-vals
                  graph-n
                  graph-rows
                  graph-vals))

(provide check-residuals
         check-twins
         column-gaps
         cosine-gaps
         edge-shares
         eigen-residuals
         embedded
         embedded-graph
         embedding-case
         embedding-checks
         graph-case
         graph-checks
         halves
         heavy-pair
         polygons
         quantised
         rows-matrix
         sign-ruled?
         split-entries
         stochastic-blocks
         stochastic-graph
         summed-entries
         thirds
         twin
         with-isolated)

(define-runtime-path twin "../python/spectral_embedding_twin.py")

(define (stochastic-blocks sizes p-in p-out seed #:weighted? [weighted? #f])
  (define truth
    (append* (for/list ([s (in-list sizes)]
                        [g (in-naturals)])
               (make-list s g))))
  (define n (length truth))
  (define groups (list->vector truth))
  (define edges
    (seeded seed
            (lambda ()
              (for*/list ([i (in-range n)]
                          [j (in-range (add1 i) n)]
                          #:when (< (random)
                                    (if (= (vector-ref groups i) (vector-ref groups j)) p-in p-out)))
                (list i
                      j
                      (if weighted?
                          (+ 0.5 (* 1.5 (random)))
                          1.0))))))
  (define both
    (sort (for*/list ([e (in-list edges)]
                      [directed (in-list (match e
                                           [(list i j w) (list e (list j i w))]))])
            directed)
          <
          #:key (match-lambda
                  [(list i j _) (+ (* i n) j)])))
  (match-define (list rows cols vals) (apply map list both))
  (values (graph rows cols vals n) truth))

(define (stochastic-graph [sizes '(40 60 80)] [seed 0] #:weighted? [weighted? #f])
  (define-values (g _truth) (stochastic-blocks sizes 0.3 0.01 seed #:weighted? weighted?))
  g)

(define (heavy-pair w #:graph [g (stochastic-graph)] #:between [ends '(0 1)])
  (match-define (list i j) ends)
  (graph-edit g
              #:rows (lambda (rows) (append rows (list i j)))
              #:cols (lambda (cols) (append cols (list j i)))
              #:vals (lambda (vals) (append vals (list w w)))))

(define (halves w)
  (list (/ w 2.0) (/ w 2.0)))

(define (thirds w)
  (define a (flsingle (/ w 3.0)))
  (list a (flsingle (- w a))))

(define (split-entries g parts)
  (define split (map parts (graph-vals g)))
  (graph-edit g
              #:rows (lambda (rows) (append rows rows))
              #:cols (lambda (cols) (append cols cols))
              #:vals (lambda (_) (append (map car split) (map cadr split)))))

(define (summed-entries g parts)
  (graph-map-vals g (lambda (w) (flsingle (apply + (parts w))))))

(define (with-isolated g count)
  (graph (graph-rows g) (graph-cols g) (graph-vals g) (+ (graph-n g) count)))

(define (polygons count m)
  (for*/list ([b (in-range count)]
              [i (in-range m)])
    (define t (/ (* 2 pi i) m))
    (list (+ (* 1000.0 b) (* 10.0 (cos t))) (* 10.0 (sin t)) 0.0)))

(define quantised
  (let-values ([(rows _labels) (centred-blobs '((0 0 0) (1 1 0) (0 1 1)) 40 7)])
    (define most (apply max (map abs (append* rows))))
    (for/list ([row (in-list rows)])
      (for/list ([x (in-list row)])
        (/ (round (* (/ x most) 1.5 (expt 2 20))) (expt 2.0 20))))))

(define (rows-matrix rows)
  (list*->device-matrix rows #:dtype 'float32))

(define (embedded result)
  (device-matrix->list* (spectral-embedding-result-embedding result)))

(define (embedded-graph g [options (hasheq)])
  (apply apply-options
         spectral-embedding-fit-graph
         options
         (append (graph->vectors g) (list (graph-n g)))))

(define (degrees g)
  (define d (make-vector (graph-n g) 0.0))
  (for ([r (in-list (graph-rows g))]
        [w (in-list (graph-vals g))])
    (vector-set! d r (+ (vector-ref d r) w)))
  d)

(define (laplacian-times g u)
  (define out (make-vector (graph-n g) 0.0))
  (for ([r (in-list (graph-rows g))]
        [c (in-list (graph-cols g))]
        [w (in-list (graph-vals g))])
    (vector-set! out r (+ (vector-ref out r) (* w (- (vector-ref u r) (vector-ref u c))))))
  out)

(define residual-bound 1e-3)

(define (column-residual g isolated mass column)
  (define u (list->vector column))
  (define lu
    (for/vector #:length (graph-n g)
                ([y (in-vector (laplacian-times g u))]
                 [x (in-vector u)]
                 [alone? (in-vector isolated)])
      (if alone? x y)))
  (define uu (for/sum ([x (in-vector u)] [m (in-vector mass)]) (* m (sqr x))))
  (define rho (/ (for/sum ([x (in-vector u)] [y (in-vector lu)]) (* x y)) uu))
  (define off
    (for/sum ([x (in-vector u)] [y (in-vector lu)] [m (in-vector mass)])
             (/ (sqr (- y (* rho m x))) m)))
  (sqrt (/ off uu)))

(define (eigen-residuals g rows #:norm-laplacian [norm #t])
  (define d (degrees g))
  (define isolated
    (for/vector ([x (in-vector d)])
      (and norm (zero? x))))
  (define mass
    (for/vector ([x (in-vector d)])
      (if (and norm (positive? x)) x 1.0)))
  (define scale
    (if norm
        1.0
        (apply max 0.0 (vector->list d))))
  (for/list ([column (in-list (transpose rows))])
    (/ (column-residual g isolated mass column) scale)))

(define (check-residuals g rows label #:norm-laplacian [norm #t])
  (define residuals (eigen-residuals g rows #:norm-laplacian norm))
  (check-true (< (apply max residuals) residual-bound) (format "~a residuals ~a" label residuals)))

(define (edge-shares g rows)
  (define d (degrees g))
  (for/list ([column (in-list (transpose rows))])
    (/ (for/sum ([(x i) (in-indexed column)] #:when (positive? (vector-ref d i))) (sqr x))
       (for/sum ([x (in-list column)]) (sqr x)))))

(define (sign-ruled? rows)
  (for/and ([column (in-list (transpose rows))])
    (> (argmax abs column) 0.0)))

(define (column-gaps ours theirs)
  (for/list ([e (in-list (transpose ours))]
             [s (in-list (transpose theirs))])
    (/ (for/fold ([gap 0.0])
                 ([a (in-list e)]
                  [b (in-list s)])
         (max gap (abs (- a b))))
       (abs (argmax abs s)))))

(define (cosine-gaps ours theirs)
  (for/list ([e (in-list (transpose ours))]
             [s (in-list (transpose theirs))])
    (- 1.0 (/ (abs (dot e s)) (sqrt (* (dot e e) (dot s s)))))))

(define (embedding-case X result #:op [op "embed"])
  (define params (spectral-embedding-result-params result))
  (hasheq 'op op
          'X X
          'k (hash-ref params 'n-components)
          'n_neighbors (spectral-embedding-result-n-neighbors result)
          'seed (hash-ref params 'random-state)
          'norm (hash-ref params 'norm-laplacian)
          'drop (hash-ref params 'drop-first)))

(define (graph-case g options #:op [op "graph"])
  (hasheq 'op op
          'rows (graph-rows g)
          'cols (graph-cols g)
          'vals (graph-vals g)
          'n (graph-n g)
          'k (hash-ref options 'n-components 2)
          'seed (hash-ref options 'random-state 0)
          'norm (hash-ref options 'norm-laplacian #t)
          'drop (hash-ref options 'drop-first #t)))

(define ((bitwise ours) theirs)
  (check-true (bits-equal? ours theirs) "embedding"))

(define (embedding-checks name rows configs)
  (define X (rows-matrix rows))
  (for/list ([options (in-list configs)])
    (define result (apply-options spectral-embedding-fit options X))
    (list (format "~a ~a" name options) (embedding-case rows result) (bitwise (embedded result)))))

(define (graph-checks name g configs)
  (for/list ([options (in-list configs)])
    (list (format "~a ~a" name options)
          (graph-case g options)
          (bitwise (embedded (embedded-graph g options))))))

(define (check-twins entries)
  (for ([entry (in-list entries)]
        [found (in-list (run-cases twin (map second entries)))])
    (match-define (list label _ check) entry)
    (test-case label
      (check (hash-ref found 'embedding)))))
