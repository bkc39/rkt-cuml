#lang racket/base

(require (only-in cuml estimator-params spectral-clustering)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list count split-at)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in threading ~>)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/python-env.rkt" run-twin test-twin twin-params)
         (only-in "private/spectral-data.rkt"
                  blobs
                  circles
                  graph-cols
                  graph-labels
                  graph-n
                  graph-rows
                  graph-vals
                  labels
                  moons
                  planted?
                  sparse-blocks))

(define-runtime-path twin "python/spectral_twin.py")

(struct spec (name data truth k options))

(define defaults
  (hasheq 'n-components #f 'random-state 0 'n-neighbors 10 'n-init 10 'eigen-tol 'auto))

(define (dense name rows truth k [options (hasheq)])
  (spec name rows truth k options))

(define (dense-specs)
  (define-values (b3 t3) (blobs 100 3 2 50.0 0))
  (define-values (b4 t4) (blobs 100 4 3 20.0 1))
  (define-values (b8 t8) (blobs 2000 5 8 8.0 7))
  (define-values (b5 t5) (blobs 2000 5 8 5.0 7))
  (define-values (c tc) (circles 1000 0.5 0.03 0))
  (define-values (m tm) (moons 1000 0.05 0))
  (list (dense "blobs 3x100" b3 t3 3)
        (dense "blobs 4x100" b4 t4 4)
        (dense "blobs 5x2000 at 8 sigma" b8 t8 5)
        (dense "blobs 5x2000 at 5 sigma" b5 t5 5)
        (dense "circles" c tc 2)
        (dense "moons" m tm 2)
        (dense "blobs 3x100, every parameter"
               b3
               t3
               3
               (hasheq 'n-components 2 'random-state 5 'n-neighbors 15 'n-init 3 'eigen-tol 1e-5))
        (dense "blobs 4x100, 8 clusters" b4 #f 8)))

(define (graph-specs)
  (append (for/list ([seed (in-range 3)])
            (define-values (g truth) (sparse-blocks '(30 40 50) 0.3 seed))
            (spec (format "blocks 30/40/50 seed ~a" seed) g truth 3 (hasheq)))
          (let-values ([(g truth) (sparse-blocks '(30 40 50) 0.3 4 #:connected? #f)])
            (list (spec "disconnected blocks" g truth 3 (hasheq 'random-state 9))))
          (let-values ([(g truth) (sparse-blocks '(300 400 500 600) 0.05 0)])
            (list (spec "blocks 300-600" g truth 4 (hasheq))))))

(define (options-of s)
  (for/fold ([o defaults]) ([(k v) (in-hash (spec-options s))])
    (hash-set o k v)))

(define (keywords-of s graph?)
  (define o
    (~> (options-of s)
        (hash-set 'n-clusters (spec-k s))
        ((lambda (o)
           (if graph?
               (hash-remove o 'n-neighbors)
               o)))))
  (define names (sort (hash-keys o) symbol<?))
  (values (map (lambda (n) (string->keyword (symbol->string n))) names)
          (map (lambda (n) (hash-ref o n)) names)))

(define (ours s graph?)
  (define-values (kws vs) (keywords-of s graph?))
  (if graph?
      (keyword-apply graph-labels kws vs (list (spec-data s)))
      (keyword-apply labels kws vs (list (mat (spec-data s) 'float32)))))

(define (python-params s)
  (define o (options-of s))
  (hasheq 'n_clusters (spec-k s)
          'n_components (or (hash-ref o 'n-components) (json-null))
          'random_state (hash-ref o 'random-state)
          'n_neighbors (hash-ref o 'n-neighbors)
          'n_init (hash-ref o 'n-init)
          'eigen_tol (if (eq? (hash-ref o 'eigen-tol) 'auto)
                         0.0
                         (hash-ref o 'eigen-tol))))

(define (dense-case s)
  (hash-set* (python-params s) 'op "fit" 'X (spec-data s)))

(define (graph-case op s)
  (define g (spec-data s))
  (hash-set* (python-params s)
             'op
             op
             'rows
             (graph-rows g)
             'cols
             (graph-cols g)
             'vals
             (graph-vals g)
             'n
             (graph-n g)))

(test-twin "T1, T2: spectral clustering's labels equal cuML's; graph partitions are sklearn's"
  (define ds (dense-specs))
  (define gs (graph-specs))
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin
              (hasheq 'cases (append (map dense-case ds)
                                     (map (lambda (s) (graph-case "precomputed" s)) gs)
                                     (map (lambda (s) (graph-case "sklearn" s)) gs)))))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  (define-values (dense-results rest) (split-at results (length ds)))
  (define-values (precomputed-results sklearn-results) (split-at rest (length gs)))
  (define dense-same
    (for/list ([s (in-list ds)]
               [theirs (in-list dense-results)])
      (define found (ours s #f))
      (test-case (spec-name s)
        (check-equal? found (hash-ref theirs 'labels) "T1 labels")
        (check-equal? (hash-ref theirs 'dtype) "int32")
        (when (spec-truth s)
          (check-true (planted? found (spec-truth s)) "the planted partition")))
      (equal? found (hash-ref theirs 'labels))))
  (define graph-outcomes
    (for/list ([s (in-list gs)]
               [theirs (in-list precomputed-results)]
               [sk (in-list sklearn-results)])
      (define found (ours s #t))
      (define agree? (planted? found (hash-ref sk 'labels)))
      (test-case (spec-name s)
        (check-equal? found (hash-ref theirs 'labels) "T1 precomputed labels")
        (check-true agree? "T2 sklearn's partition")
        (check-true (planted? found (spec-truth s)) "the planted partition"))
      (list (equal? found (hash-ref theirs 'labels)) agree?)))
  (printf "twin summary: ~a dense and ~a graph spectral cases; labels equal cuML's in ~a and ~a\n"
          (length ds)
          (length gs)
          (count values dense-same)
          (count car graph-outcomes))
  (printf "twin sklearn: ~a graphs; partitions agree in ~a\n"
          (length gs)
          (count cadr graph-outcomes)))

(test-twin "T3: cuml's SpectralClustering default parameters are the estimator's"
  (match-define (hash-table ('results (list theirs)))
    (run-twin twin (hasheq 'cases (list (hasheq 'op "params")))))
  (define mapped (hash-update (twin-params theirs '()) 'random-state (lambda (seed) (or seed 0))))
  (define ours-params (~> (spectral-clustering) estimator-params (hash-remove 'resources)))
  (printf "twin params: ~s\n" theirs)
  (check-equal? (sort (hash-keys mapped) symbol<?) (sort (hash-keys ours-params) symbol<?))
  (check-equal? mapped ours-params))
