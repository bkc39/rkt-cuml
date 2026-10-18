#lang racket/base

(require (only-in cuml/cluster
                  kmeans-fit
                  kmeans-predict
                  kmeans-result
                  kmeans-score
                  kmeans-transform)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list take)
         (only-in racket/match match match-define)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in "private/kmeans-data.rkt" blob-data host host-vector mat squared-distance vec)
         (only-in "private/python-env.rkt" close? run-twin test-twin))

(define-runtime-path twin "python/kmeans_twin.py")

(define separated
  '((fit 300 2 3 float32 k-means++ 1)
    (fit 500 5 4 float32 scalable-k-means++ 2)
    (fit 400 3 5 float32 random 3)
    (fit 300 4 3 float32 array 4)
    (fit 300 2 3 float64 k-means++ 5)
    (fit 500 8 6 float64 scalable-k-means++ 6)
    (fit 1000 16 8 float32 k-means++ 7)
    (fit 250 3 2 float64 random 8)))

(define overlapping
  (for*/list ([init (in-list '(scalable-k-means++ k-means++ random))]
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
    [(list* key value rest) (hash-set (options->hash rest) (string->symbol (keyword->string key)) value)]))

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
                #:init (if init-rows (mat init-rows (spec-dt s)) (spec-init s))
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

(define deviations (make-hasheq))
(define off-by-one '())

(define (note! dt deviation)
  (hash-update! deviations dt (lambda (m) (max m deviation)) 0.0))

(define (compare-fit r theirs)
  (match-define (run s _ _ _ (kmeans-result centroids labels inertia n-iter)) r)
  (define ours-labels (host-vector labels))
  (define ours-centroids (host centroids))
  (define their-iter (hash-ref theirs 'n_iter))
  (define deviation (max-deviation ours-centroids (hash-ref theirs 'centroids)))
  (define inertia-deviation (relative-deviation inertia (hash-ref theirs 'inertia)))
  (define ari (hash-ref theirs 'ari))
  (note! (spec-dt s) (max deviation inertia-deviation))
  (unless (= n-iter their-iter)
    (set! off-by-one (cons (spec-name s) off-by-one)))
  (printf "twin ~a: n_iter ~a/~a, inertia ~a/~a, centroid deviation ~a, ARI ~a\n"
          (spec-name s)
          n-iter
          their-iter
          inertia
          (hash-ref theirs 'inertia)
          deviation
          ari)
  (define exact-iter? (and (eq? (spec-init s) 'array) (memv (option s 'max-iter 300) '(0 1))))
  (match (list (spec-dt s) (spec-kind s))
    [(list 'float64 _)
     (check-equal? ours-labels (hash-ref theirs 'labels))
     (check-equal? n-iter their-iter)
     (check-true (<= deviation 1e-9) "centroids within 1e-9")
     (check-true (<= inertia-deviation 1e-9) "inertia within 1e-9")]
    [(list 'float32 'overlapping)
     (check-true (<= (abs (- n-iter their-iter)) 1) "n_iter within 1")
     (check-true (if (= n-iter their-iter) (= ari 1.0) (>= ari 0.99)) (format "ARI ~a" ari))
     (check-true (<= inertia-deviation 1e-4) "inertia within 1e-4")]
    [(list 'float32 _)
     (check-equal? ours-labels (hash-ref theirs 'labels))
     (check-true (<= deviation 1e-4) "centroids within 1e-4")
     (check-true (<= inertia-deviation 1e-4) "inertia within 1e-4")
     (if exact-iter?
         (check-equal? n-iter their-iter)
         (check-true (<= (abs (- n-iter their-iter)) 1) "n_iter within 1"))]))

(define (norm2 v)
  (for/sum ([a (in-list v)])
    (* a a)))

(define (compare-apply a theirs)
  (match-define (applied s Y C _ labels transform score weighted-score) a)
  (define tolerance (if (eq? (spec-dt s) 'float32) 1e-5 1e-12))
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
      (check-true (<= (abs (- d e)) (* tolerance (max 1.0 (+ (norm2 y) (norm2 c)))))
                  (format "transform ~a against the twin's ~a" d e))))
  (check-true (close? score (hash-ref theirs 'score) tolerance) "score")
  (check-true (close? weighted-score (hash-ref theirs 'weighted_score) tolerance) "weighted score"))

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
  (for ([r (in-list runs)]
        [theirs (in-list results)])
    (test-case (spec-name (run-spec r))
      (compare-fit r theirs)))
  (for ([a (in-list appliers)]
        [theirs (in-list (list-tail results (length runs)))])
    (test-case (format "apply ~a" (spec-name (applied-spec a)))
      (compare-apply a theirs)))
  (printf "twin summary: ~a fit and ~a apply cases; largest relative deviation float32 ~a, float64 ~a; n_iter off by one in ~a\n"
          (length runs)
          (length appliers)
          (hash-ref deviations 'float32 0.0)
          (hash-ref deviations 'float64 0.0)
          (if (null? off-by-one) "none" (reverse off-by-one))))
