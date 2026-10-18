#lang racket/base

(require (only-in cuml/cluster kmeans-fit kmeans-predict kmeans-result kmeans-score kmeans-transform)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list take)
         (only-in racket/match match match* match-define)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in rackunit check-equal? check-not-equal? check-true test-case)
         (only-in "private/kmeans-data.rkt"
                  blob-data
                  close?
                  dtypes
                  host
                  host-vector
                  mat
                  norm2
                  scaled-close?
                  string-inits
                  tolerance
                  vec)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/kmeans_twin.py")

(define separated
  '((fit 300 2 3 float32 k-means++ 1) (fit 500 5 4 float32 scalable-k-means++ 2)
                                      (fit 400 3 5 float32 random 3)
                                      (fit 300 4 3 float32 array 4)
                                      (fit 300 2 3 float64 k-means++ 5)
                                      (fit 500 8 6 float64 scalable-k-means++ 6)
                                      (fit 1000 16 8 float32 k-means++ 7)
                                      (fit 250 3 2 float64 random 8)))

(define overlapping
  (for*/list ([init (in-list string-inits)]
              [seed (in-list '(1 2 3))])
    `(overlapping 2000 4 8 float32 ,init ,seed)))

(define weighted
  '((fit-weighted 600 3 4 float32 scalable-k-means++ 11) (fit-weighted 600 3 4 float64 random 12)))

(define parameters
  '((fit 400 2 4 float64 random 13 #:n-init 3)
    (fit 300 3 3 float64 array 14 #:max-iter 0)
    (fit 300 3 3 float64 array 14 #:max-iter 1)
    (fit 300 3 3 float32 array 14 #:max-iter 0)
    (fit 300 3 3 float32 array 14 #:max-iter 1)
    (fit 1000 4 5 float32 scalable-k-means++ 15 #:max-samples-per-batch 64)
    (fit 500 4 6 float32 scalable-k-means++ 16 #:oversampling-factor 0.5)
    (fit 500 3 4 float64 random 17 #:tol 1e-2)))

(define (weights n)
  (parameterize ([current-pseudo-random-generator
                  (vector->pseudo-random-generator (vector 11 22 33 44 55 66))])
    (for/list ([_ (in-range n)])
      (add1 (random 5)))))

(define (options->hash options)
  (match options
    ['() (hasheq)]
    [(list* key value rest)
     (hash-set (options->hash rest) (string->symbol (keyword->string key)) value)]))

(struct spec (kind n d k dt init seed options))

(define (parse s)
  (match-define (list* kind n d k dt init seed options) s)
  (spec kind n d k dt init seed (options->hash options)))

(define (option s key default)
  (hash-ref (spec-options s) key default))

(define (spec-name s)
  (format "~a ~ax~a k~a ~a ~a seed ~a~a"
          (spec-kind s)
          (spec-n s)
          (spec-d s)
          (spec-k s)
          (spec-dt s)
          (spec-init s)
          (spec-seed s)
          (for/fold ([text ""]) ([(key value) (in-hash (spec-options s))])
            (format "~a ~a=~a" text key value))))

(define (spec-data s)
  (match (spec-kind s)
    ['overlapping (blob-data (spec-n s) (spec-d s) (spec-k s) 42 (spec-dt s) #:std 4.0)]
    [_ (blob-data (spec-n s) (spec-d s) (spec-k s) (+ (spec-seed s) 100) (spec-dt s))]))

(define (or-null v)
  (or v (json-null)))

(struct run (spec X init-rows w result))

(define (racket-fit s)
  (define X (spec-data s))
  (define init-rows (and (eq? (spec-init s) 'array) (take (host X) (spec-k s))))
  (define w (and (eq? (spec-kind s) 'fit-weighted) (weights (spec-n s))))
  (define result
    (kmeans-fit X
                #:n-clusters (spec-k s)
                #:init (if init-rows
                           (mat init-rows (spec-dt s))
                           (spec-init s))
                #:random-state (spec-seed s)
                #:max-iter (option s 'max-iter 300)
                #:tol (option s 'tol 1e-4)
                #:n-init (option s 'n-init 'auto)
                #:oversampling-factor (option s 'oversampling-factor 2.0)
                #:max-samples-per-batch (option s 'max-samples-per-batch 32768)
                #:sample-weight (and w (vec w (spec-dt s)))))
  (run s X init-rows w result))

(define (fit-case r)
  (match-define (run s X init-rows w (kmeans-result _ labels _ _)) r)
  (hasheq 'op "fit"
          'dtype (symbol->string (spec-dt s))
          'X (host X)
          'n_clusters (spec-k s)
          'init (or init-rows (symbol->string (spec-init s)))
          'n_init (match (option s 'n-init 'auto)
                    ['auto "auto"]
                    [n n])
          'max_iter (option s 'max-iter 300)
          'tol (option s 'tol 1e-4)
          'random_state (spec-seed s)
          'oversampling_factor (option s 'oversampling-factor 2.0)
          'max_samples_per_batch (option s 'max-samples-per-batch 32768)
          'sample_weight (or-null w)
          'labels (host-vector labels)))

(struct applied (spec Y C w labels transform score weighted-score))

(define (racket-apply r)
  (match-define (run s _ _ _ (kmeans-result C _ _ _)) r)
  (define Y (blob-data 200 (spec-d s) (spec-k s) (+ (spec-seed s) 200) (spec-dt s)))
  (define w (weights 200))
  (define wv (vec w (spec-dt s)))
  (applied s
           Y
           C
           w
           (host-vector (kmeans-predict C Y))
           (host (kmeans-transform C Y))
           (kmeans-score C Y)
           (kmeans-score C Y #:sample-weight wv)))

(define (apply-case a)
  (hasheq 'op "apply"
          'dtype (symbol->string (spec-dt (applied-spec a)))
          'Y (host (applied-Y a))
          'centroids (host (applied-C a))
          'sample_weight (applied-w a)))

(define (relative-deviation actual expected)
  (/ (abs (- actual expected)) (max 1.0 (abs expected))))

(define (max-deviation actual-rows expected-rows)
  (for*/fold ([m 0.0])
             ([(a-row e-row) (in-parallel (in-list actual-rows) (in-list expected-rows))]
              [(a e) (in-parallel (in-list a-row) (in-list e-row))])
    (max m (relative-deviation a e))))

(struct fitted (labels centroids inertia n-iter))

(define (run-fitted r)
  (match-define (run _ _ _ _ (kmeans-result centroids labels inertia n-iter)) r)
  (fitted (host-vector labels) (host centroids) inertia n-iter))

(define (matched-centroids labels their-labels their-centroids)
  (define theirs (list->vector their-centroids))
  (define pairing
    (for/hash ([ours (in-list labels)]
               [their (in-list their-labels)])
      (values ours their)))
  (for/list ([i (in-range (vector-length theirs))])
    (vector-ref theirs (hash-ref pairing i i))))

(struct comparison (dt deviation off-by-one? problems))

(define (problem ok? message)
  (if ok?
      '()
      (list message)))

(define (compare-fit s ours theirs)
  (match-define (fitted labels centroids inertia n-iter) ours)
  (define tol (if (eq? (spec-dt s) 'float32) 1e-4 1e-9))
  (define their-centroids (hash-ref theirs 'centroids))
  (define iter-gap (abs (- n-iter (hash-ref theirs 'n_iter))))
  (define ari (hash-ref theirs 'ari))
  (define inertia-deviation (relative-deviation inertia (hash-ref theirs 'inertia)))
  (define matched-deviation
    (max-deviation centroids (matched-centroids labels (hash-ref theirs 'labels) their-centroids)))
  (define exact-iter? (and (eq? (spec-init s) 'array) (memv (option s 'max-iter 300) '(0 1))))
  (define random-restarts? (and (eq? (spec-init s) 'random) (not (eqv? (option s 'n-init 'auto) 1))))
  (define problems
    (append (problem (<= inertia-deviation tol) (format "inertia within ~a" tol))
            (match* ((spec-dt s) (spec-kind s))
              [('float32 'overlapping)
               (append (problem (<= iter-gap 1) "n_iter within 1")
                       (problem (if (zero? iter-gap)
                                    (= ari 1.0)
                                    (>= ari 0.99))
                                (format "ARI ~a" ari)))]
              [(_ _)
               #:when random-restarts?
               (append (problem (= ari 1.0) (format "ARI ~a" ari))
                       (problem (<= matched-deviation tol)
                                (format "centroids matched by label within ~a" tol)))]
              [(dt _)
               (append (problem (equal? labels (hash-ref theirs 'labels)) "labels")
                       (problem (<= (max-deviation centroids their-centroids) tol)
                                (format "centroids within ~a" tol))
                       (if (or (eq? dt 'float64) exact-iter?)
                           (problem (zero? iter-gap) "n_iter equal")
                           (problem (<= iter-gap 1) "n_iter within 1")))])))
  (comparison (spec-dt s) (max matched-deviation inertia-deviation) (positive? iter-gap) problems))

(define (compare-apply a theirs)
  (match-define (applied s Y C _ labels transform score weighted-score) a)
  (define tol (tolerance (spec-dt s)))
  (define cs (host C))
  (printf "twin apply ~a: score ~a/~a, weighted score ~a/~a\n"
          (spec-name s)
          score
          (hash-ref theirs 'score)
          weighted-score
          (hash-ref theirs 'weighted_score))
  (check-equal? labels (hash-ref theirs 'labels))
  (for ([y (in-list (host Y))]
        [ours (in-list transform)]
        [their-row (in-list (hash-ref theirs 'transform))])
    (for ([c (in-list cs)]
          [d (in-list ours)]
          [e (in-list their-row)])
      (check-true (scaled-close? d e (+ (norm2 y) (norm2 c)) tol)
                  (format "transform ~a against the twin's ~a" d e))))
  (check-true (close? score (hash-ref theirs 'score) tol) "score")
  (check-true (close? weighted-score (hash-ref theirs 'weighted_score) tol) "weighted score"))

(test-case "compare-fit takes a random-restart answer that permutes the clusters, not a moved centroid"
  (define ours (fitted '(0 0 1 1 2) '((0.0 0.0) (5.0 5.0) (9.0 1.0)) 10.0 4))
  (define (answer last-row)
    (hasheq 'labels '(2 2 0 0 1)
            'centroids (list '(5.0 5.0) last-row '(0.0 0.0))
            'inertia 10.0
            'n_iter 5
            'ari 1.0))
  (for ([dt (in-list dtypes)]
        [moved (in-list '(1.001 1.00000001))])
    (define s (parse `(fit 400 3 5 ,dt random 3)))
    (check-equal? (comparison-problems (compare-fit s ours (answer '(9.0 1.0)))) '())
    (check-not-equal? (comparison-problems (compare-fit s ours (answer (list 9.0 moved)))) '())))

(test-case "compare-fit pairs a random-restart answer's centroids by label, not by sort order"
  (for ([dt (in-list dtypes)]
        [near (in-list '(1.00001 1.0000000001))]
        [moved (in-list '(0.001 0.00000001))])
    (define s (parse `(fit 400 3 5 ,dt random 3)))
    (define ours (fitted '(0 0 1 1) (list '(1.0 5.0) (list near 0.0)) 10.0 4))
    (define (answer last-row)
      (hasheq 'labels '(0 0 1 1)
              'centroids (list (list near 5.0) last-row)
              'inertia 10.0
              'n_iter 5
              'ari 1.0))
    (check-equal? (comparison-problems (compare-fit s ours (answer '(1.0 0.0)))) '())
    (check-not-equal? (comparison-problems (compare-fit s ours (answer (list 1.0 moved)))) '())))

(test-twin "kmeans-fit, predict, transform and score match cuml.KMeans on the same data and seeds"
  (define specs (map parse (append separated overlapping weighted parameters)))
  (define runs (map racket-fit specs))
  (define appliers
    (for/list ([r (in-list runs)]
               #:when (memq (spec-kind (run-spec r)) '(fit-weighted fit))
               #:when (hash-empty? (spec-options (run-spec r))))
      (racket-apply r)))
  (define answer
    (run-twin twin (hasheq 'cases (append (map fit-case runs) (map apply-case appliers)))))
  (match-define (hash-table ('results results) ('version version)) answer)
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  (check-equal? (list (length runs) (length appliers)) '(27 10))
  (define comparisons
    (for/list ([r (in-list runs)]
               [theirs (in-list results)])
      (define s (run-spec r))
      (define ours (run-fitted r))
      (define c (compare-fit s ours theirs))
      (printf "twin ~a: n_iter ~a/~a, inertia ~a/~a, centroid deviation ~a, ARI ~a\n"
              (spec-name s)
              (fitted-n-iter ours)
              (hash-ref theirs 'n_iter)
              (fitted-inertia ours)
              (hash-ref theirs 'inertia)
              (comparison-deviation c)
              (hash-ref theirs 'ari))
      (test-case (spec-name s)
        (check-equal? (comparison-problems c) '()))
      c))
  (define (largest dt)
    (for/fold ([m 0.0])
              ([c (in-list comparisons)]
               #:when (eq? (comparison-dt c) dt))
      (max m (comparison-deviation c))))
  (define off-by-one
    (for/list ([r (in-list runs)]
               [c (in-list comparisons)]
               #:when (comparison-off-by-one? c))
      (spec-name (run-spec r))))
  (for ([a (in-list appliers)]
        [theirs (in-list (list-tail results (length runs)))])
    (test-case (format "apply ~a" (spec-name (applied-spec a)))
      (compare-apply a theirs)))
  (printf
   "twin summary: ~a fit and ~a apply cases; largest relative deviation float32 ~a, float64 ~a; n_iter off by one in ~a\n"
   (length runs)
   (length appliers)
   (largest 'float32)
   (largest 'float64)
   (if (null? off-by-one) "none" off-by-one)))
