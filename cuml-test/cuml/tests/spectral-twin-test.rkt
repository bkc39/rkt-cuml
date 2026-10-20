#lang racket/base

(require (only-in cuml estimator-params spectral-clustering)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list count split-at)
         (only-in racket/match match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in threading cond~> ~>)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/python-env.rkt" run-twin test-twin twin-params)
         (only-in "private/spectral-data.rkt"
                  blobs
                  circles
                  fit-case-compare
                  fit-case-data
                  fit-case-k
                  fit-case-name
                  fit-case-options
                  fit-case-truth
                  graph-cols
                  graph-labels
                  graph-n
                  graph-rows
                  graph-vals
                  labels
                  moons
                  planted
                  planted?
                  sparse-blocks))

(define-runtime-path twin "python/spectral_twin.py")

(define defaults
  (hasheq 'n-components #f 'random-state 0 'n-neighbors 10 'n-init 10 'eigen-tol 'auto))

(define (dense-specs)
  (define-values (b3 t3) (blobs 100 3 2 50.0 0))
  (define-values (b4 t4) (blobs 100 4 3 20.0 1))
  (define-values (b8 t8) (blobs 2000 5 8 8.0 7))
  (define-values (b5 t5) (blobs 2000 5 8 5.0 7))
  (define-values (c tc) (circles 1000 0.5 0.03 0))
  (define-values (m tm) (moons 1000 0.05 0))
  (list (planted "blobs 3x100" b3 t3 3)
        (planted "blobs 4x100" b4 t4 4)
        (planted "blobs 5x2000 at 8 sigma" b8 t8 5)
        (planted "blobs 5x2000 at 5 sigma" b5 t5 5)
        (planted "circles" c tc 2)
        (planted "moons" m tm 2)
        (planted "blobs 3x100, every parameter"
                 b3
                 t3
                 3
                 #:options
                 (hasheq 'n-components 2 'random-state 5 'n-neighbors 15 'n-init 3 'eigen-tol 1e-5))
        (planted "blobs 4x100, 8 clusters, one init" b4 #f 8 #:options (hasheq 'n-init 1))))

(define (graph-specs)
  (append (for/list ([seed (in-range 3)])
            (define-values (g truth) (sparse-blocks '(30 40 50) 0.3 seed))
            (planted (format "blocks 30/40/50 seed ~a" seed) g truth 3))
          (let-values ([(g truth) (sparse-blocks '(30 40 50) 0.3 4 #:connected? #f)])
            (list (planted "disconnected blocks" g truth 3 #:options (hasheq 'random-state 9))))
          (let-values ([(g truth) (sparse-blocks '(300 400 500 600) 0.05 0)])
            (list (planted "blocks 300-600" g truth 4 #:compare 'partition)))))

(define (options-of s)
  (for/fold ([options defaults]) ([(k v) (in-hash (fit-case-options s))])
    (hash-set options k v)))

(define (ours s graph?)
  (define options
    (cond~> (hash-set (options-of s) 'n-clusters (fit-case-k s)) [graph? (hash-remove 'n-neighbors)]))
  (if graph?
      (apply-options graph-labels options (fit-case-data s))
      (apply-options labels options (mat (fit-case-data s) 'float32))))

(define (python-params s)
  (define o (options-of s))
  (hasheq 'n_clusters (fit-case-k s)
          'n_components (or (hash-ref o 'n-components) (json-null))
          'random_state (hash-ref o 'random-state)
          'n_neighbors (hash-ref o 'n-neighbors)
          'n_init (hash-ref o 'n-init)
          'eigen_tol (if (eq? (hash-ref o 'eigen-tol) 'auto)
                         0.0
                         (hash-ref o 'eigen-tol))))

(define (dense-case s)
  (hash-set* (python-params s) 'op "fit" 'X (fit-case-data s)))

(define (graph-case op s)
  (define g (fit-case-data s))
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
      (test-case (fit-case-name s)
        (check-equal? found (hash-ref theirs 'labels) "T1 labels")
        (check-equal? (hash-ref theirs 'dtype) "int32")
        (when (fit-case-truth s)
          (check-true (planted? found (fit-case-truth s)) "the planted partition")))
      (equal? found (hash-ref theirs 'labels))))
  (define graph-outcomes
    (for/list ([s (in-list gs)]
               [theirs (in-list precomputed-results)]
               [sk (in-list sklearn-results)])
      (define found (ours s #t))
      (define cuml-labels (hash-ref theirs 'labels))
      (define agree? (planted? found (hash-ref sk 'labels)))
      (test-case (fit-case-name s)
        (match (fit-case-compare s)
          ['labels (check-equal? found cuml-labels "T1 precomputed labels")]
          ['partition (check-true (planted? found cuml-labels) "T1 precomputed partition")])
        (check-true agree? "T2 sklearn's partition")
        (check-true (planted? found (fit-case-truth s)) "the planted partition"))
      (list (match (fit-case-compare s)
              ['labels (equal? found cuml-labels)]
              ['partition (planted? found cuml-labels)])
            agree?)))
  (printf
   "twin summary: ~a dense and ~a graph spectral cases; labels equal cuML's in ~a and ~a (~a graph by partition)\n"
   (length ds)
   (length gs)
   (count values dense-same)
   (count car graph-outcomes)
   (count (lambda (s) (eq? (fit-case-compare s) 'partition)) gs))
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
