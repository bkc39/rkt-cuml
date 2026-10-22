#lang racket/base

(require (only-in cuml/cluster dbscan-fit dbscan-params dbscan-result-labels)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/match == match match* match-define)
         racket/runtime-path
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix->list* dtype)
         (only-in threading ~>)
         (only-in "private/dbscan-data.rkt"
                  cosine-cloud
                  distance-matrix
                  integer-grid
                  mat
                  result->lists
                  seeded
                  vec)
         (only-in "private/kmeans-data.rkt" blob-data)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/dbscan_twin.py")

(struct spec (kind dt eps min-samples metric out-dtype mbytes))

(define specs
  (for*/list ([dt (in-list '(float32 float64))]
              [s (in-list '((blob 0.5 5 euclidean int32 #f)
                            (grid 1.5 4 euclidean int32 #f)
                            (weighted 1.5 6 euclidean int32 #f)
                            (precomputed 1.5 4 precomputed int32 #f)
                            (cosine 0.001953125 4 cosine int32 #f)
                            (onehot 1.0 10 cosine int32 #f)
                            (weighted-cosine 0.001953125 8 cosine int32 #f)
                            (batched 1.5 4 euclidean int32 1)
                            (grid 1.5 4 euclidean int64 #f)
                            (batched 1.5 4 euclidean int64 2)))])
    (match-define (list kind eps min-samples metric out-dtype mbytes) s)
    (spec kind dt eps min-samples metric out-dtype mbytes)))

(define (spec-name s)
  (format "~a ~a eps ~a min-samples ~a ~a ~a max-mbytes-per-batch ~a"
          (spec-kind s)
          (spec-dt s)
          (spec-eps s)
          (spec-min-samples s)
          (spec-metric s)
          (spec-out-dtype s)
          (spec-mbytes s)))

(define (rows-of s)
  (match (spec-kind s)
    ['blob (device-matrix->list* (blob-data 1000 3 4 7 (spec-dt s) #:std 0.4))]
    ['grid (integer-grid 500 2 25 1)]
    ['weighted (integer-grid 400 2 20 2)]
    ['precomputed (distance-matrix (integer-grid 300 2 30 3))]
    [(or 'cosine 'weighted-cosine) (cosine-cloud 300 3 (spec-eps s) 4)]
    ['onehot
     (seeded 11
             (lambda ()
               (for/list ([_ (in-range 40)])
                 (define hot (random 8))
                 (for/list ([j (in-range 8)])
                   (if (= j hot) 1 0)))))]
    ['batched (integer-grid 2001 2 60 5)]))

(define (weights-of s n)
  (match (spec-kind s)
    [(or 'weighted 'weighted-cosine)
     (seeded 6
             (lambda ()
               (for/list ([_ (in-range n)])
                 (add1 (random 3)))))]
    [_ #f]))

(struct run (spec rows weights found labels-dtype))

(define (racket-fit s)
  (define rows (rows-of s))
  (define w (weights-of s (length rows)))
  (define r
    (dbscan-fit (mat rows (spec-dt s))
                #:eps (spec-eps s)
                #:min-samples (spec-min-samples s)
                #:metric (spec-metric s)
                #:out-dtype (spec-out-dtype s)
                #:max-mbytes-per-batch (spec-mbytes s)
                #:sample-weight (and w (vec w (spec-dt s)))))
  (run s rows w (result->lists r) (dtype (dbscan-result-labels r))))

(define (or-null v)
  (or v (json-null)))

(define (sklearn? s)
  (eq? (spec-metric s) 'cosine))

(define (fit-case r)
  (match-define (run s rows w _ _) r)
  (hasheq 'op (if (sklearn? s) "sklearn_fit" "fit")
          'dtype (symbol->string (spec-dt s))
          'X (map (lambda (row) (map exact->inexact row)) rows)
          'eps (spec-eps s)
          'min_samples (spec-min-samples s)
          'metric (symbol->string (spec-metric s))
          'out_dtype (symbol->string (spec-out-dtype s))
          'max_mbytes_per_batch (or-null (spec-mbytes s))
          'sample_weight (or-null w)))

(test-twin "dbscan-fit's labels and core samples equal cuml.DBSCAN's, and sklearn's under cosine"
  (define runs (map racket-fit specs))
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases (map fit-case runs))))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  (define equal-count
    (for/sum ([r (in-list runs)] [theirs (in-list results)])
             (match-define (run s _ _ (list labels core) labels-dtype) r)
             (define out (symbol->string (spec-out-dtype s)))
             (define same?
               (equal? (list labels core) (list (hash-ref theirs 'labels) (hash-ref theirs 'core))))
             (printf "twin ~a: ~a clusters, ~a core samples, ~a noise; ~a\n"
                     (spec-name s)
                     (add1 (apply max -1 labels))
                     (length core)
                     (for/sum ([l (in-list labels)]) (if (= l -1) 1 0))
                     (if same? "equal" "DIFFERENT"))
             (test-case (spec-name s)
               (check-equal? labels (hash-ref theirs 'labels) "labels")
               (check-equal? core (hash-ref theirs 'core) "core samples")
               (check-equal? (symbol->string labels-dtype) out "labels dtype")
               (unless (sklearn? s)
                 (check-equal? (hash-ref theirs 'dtypes) (list out out))))
             (if same? 1 0)))
  (printf "twin summary: ~a DBSCAN cases; labels and core samples equal in ~a\n"
          (length runs)
          equal-count))

(define unbound-twin-params '(verbose output-type algorithm calc-core-sample-indices))

(define (twin-key key)
  (~> key symbol->string (regexp-replace* #rx"_" _ "-") string->symbol))

(define (twin-value key value)
  (match* (key value)
    [(_ (== (json-null))) #f]
    [(_ (? string?)) (string->symbol value)]
    [(_ _) value]))

(test-twin "cuml.DBSCAN's default parameters are dbscan-params'"
  (match-define (hash-table ('results (list theirs)))
    (run-twin twin (hasheq 'cases (list (hasheq 'op "params")))))
  (define mapped
    (for*/hasheq ([(key value) (in-hash theirs)]
                  [name (in-value (twin-key key))]
                  #:unless (memq name unbound-twin-params))
      (values name (twin-value name value))))
  (define ours (hash-remove (dbscan-params) 'out-dtype))
  (printf "twin params: ~s\n" theirs)
  (check-equal? mapped ours))
