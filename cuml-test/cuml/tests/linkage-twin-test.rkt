#lang racket/base

(require (only-in cuml agglomerative-clustering estimator-params)
         (only-in cuml/cluster single-linkage-fit single-linkage-result-labels)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/flonum flsingle)
         (only-in racket/list range split-at)
         (only-in racket/match == match match* match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* dtype)
         (only-in threading ~>)
         (only-in "private/dbscan-data.rkt" integer-grid mat same-partition?)
         (only-in "private/kmeans-data.rkt" blob-data)
         (only-in "private/linkage-data.rkt"
                  hand-rows
                  linkage-oracle
                  oracle-partition
                  permuted-rows
                  result->lists
                  two-far)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/linkage_twin.py")

(struct spec (name rows ks exact?))

(define (float32-rows rows)
  (for/list ([r (in-list rows)])
    (for/list ([x (in-list r)])
      (flsingle (exact->inexact x)))))

(define (specs)
  (list (spec "hand grid" hand-rows (range 1 6) #t)
        (spec "permuted hand grid" permuted-rows (range 1 6) #t)
        (spec "integer grid 500x2" (integer-grid 500 2 30 1) '(2 5 9) #t)
        (spec "integer grid 300x3" (integer-grid 300 3 10 2) '(3 7) #t)
        (spec "blobs" (device-matrix->list* (blob-data 1000 3 4 7 'float32)) '(4 6) #f)
        (spec "two far blobs" (two-far 3) '(2 3) #f)
        (spec "two rows" '((1 2) (5 3)) '(1 2) #t)
        (spec "k 1 and k n" (integer-grid 30 2 100 4) '(1 30) #t)))

(struct run (spec connectivity k labels children labels-dtype))

(define (racket-fit s connectivity k)
  (define r
    (single-linkage-fit (mat (spec-rows s) 'float32) #:n-clusters k #:connectivity connectivity))
  (match-define (list labels children) (result->lists r))
  (run s connectivity k labels children (dtype (single-linkage-result-labels r))))

(define (runs)
  (for*/list ([s (in-list (specs))]
              [connectivity (in-list '(pairwise knn))]
              [k (in-list (spec-ks s))])
    (racket-fit s connectivity k)))

(define (run-name r)
  (format "~a ~a k ~a" (spec-name (run-spec r)) (run-connectivity r) (run-k r)))

(define (case-of op r)
  (hasheq 'op op
          'X (map (lambda (row) (map exact->inexact row)) (spec-rows (run-spec r)))
          'n_clusters (run-k r)
          'connectivity (symbol->string (run-connectivity r))
          'c 15
          'children (run-children r)))

(define (unique-cut r)
  (define rows (spec-rows (run-spec r)))
  (oracle-partition (linkage-oracle (if (spec-exact? (run-spec r))
                                        rows
                                        (float32-rows rows)))
                    (run-k r)))

(test-twin "single-linkage-fit's labels and children equal cuML's; its tree is scipy's; its cuts sklearn's"
  (define all (runs))
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin
              (hasheq 'cases (append (map (lambda (r) (case-of "fit" r)) all)
                                     (map (lambda (r) (case-of "scipy" r)) all)
                                     (map (lambda (r) (case-of "sklearn" r)) all)))))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  (define count (length all))
  (define-values (cuml-results rest) (split-at results count))
  (define-values (scipy-results sklearn-results) (split-at rest count))
  (define-values (equal-count exact-count worst agree-count cut-count)
    (for/fold ([equal-count 0]
               [exact-count 0]
               [worst 0.0]
               [agree-count 0]
               [cut-count 0])
              ([r (in-list all)]
               [theirs (in-list cuml-results)]
               [scipy (in-list scipy-results)]
               [sk (in-list sklearn-results)])
      (match-define (run s _ _ labels children labels-dtype) r)
      (match-define (hash-table ('worst error) ('largest largest) ('pairs pairs)) scipy)
      (define relative
        (if (zero? largest)
            0.0
            (/ error largest)))
      (define cut (unique-cut r))
      (define n (length labels))
      (define agree? (and cut (same-partition? labels (range n) (hash-ref sk 'labels) (range n))))
      (define same?
        (equal? (list labels children) (list (hash-ref theirs 'labels) (hash-ref theirs 'children))))
      (test-case (run-name r)
        (check-equal? labels (hash-ref theirs 'labels) "T1 labels")
        (check-equal? children (hash-ref theirs 'children) "T1 children")
        (check-equal? (hash-ref theirs 'dtype) "int32")
        (check-equal? labels-dtype 'int32)
        (if (spec-exact? s)
            (check-equal? error 0.0 "T2 cophenetic, exactly")
            (check-true (<= relative 1e-5) (format "T2 cophenetic, relative error ~a" relative)))
        (when (member (spec-name s) '("hand grid" "permuted hand grid"))
          (check-equal? children pairs "T2 children are scipy's pairs"))
        (when cut
          (check-true agree? "T3 sklearn's partition")))
      (values (+ equal-count (if same? 1 0))
              (+ exact-count (if (zero? error) 1 0))
              (max worst relative)
              (+ agree-count (if agree? 1 0))
              (+ cut-count (if cut 1 0)))))
  (printf "twin summary: ~a single-linkage cases; labels and children equal cuML's in ~a\n"
          count
          equal-count)
  (printf "twin scipy: ~a cases; cophenetic exact in ~a, worst relative error ~a\n"
          count
          exact-count
          worst)
  (printf "twin sklearn: ~a unique cuts; partitions agree in ~a\n" cut-count agree-count))

(define unbound-twin-params '(verbose output-type handle))

(define (twin-key key)
  (~> key symbol->string (regexp-replace* #rx"_" _ "-") string->symbol))

(define (twin-value key value)
  (match* (key value)
    [(_ (== (json-null))) #f]
    [(_ (? string?)) (string->symbol value)]
    [(_ _) value]))

(test-twin "T4: cuml.AgglomerativeClustering's default parameters are the estimator's"
  (match-define (hash-table ('results (list theirs)))
    (run-twin twin (hasheq 'cases (list (hasheq 'op "params")))))
  (define mapped
    (for*/hasheq ([(key value) (in-hash theirs)]
                  [name (in-value (twin-key key))]
                  #:unless (memq name unbound-twin-params))
      (values name (twin-value name value))))
  (define ours (~> (agglomerative-clustering) estimator-params (hash-remove 'resources)))
  (printf "twin params: ~s\n" theirs)
  (check-equal? mapped ours))
