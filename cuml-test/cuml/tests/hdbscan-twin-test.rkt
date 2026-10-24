#lang racket/base

(require (only-in cuml estimator-params hdbscan)
         (only-in cuml/cluster
                  hdbscan-all-points-membership-vectors
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-membership-vector
                  hdbscan-params)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list count split-at)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/estimator-data.rkt" apply-options hash-remove*)
         (only-in "private/hdbscan-data.rkt"
                  alternate
                  close?
                  dataset
                  fit-lists
                  fitted-children
                  fitted-destinations
                  fitted-distances
                  fitted-labels
                  fitted-lambdas
                  fitted-linkage
                  fitted-linkage-sizes
                  fitted-parents
                  fitted-persistence
                  fitted-probabilities
                  fitted-sizes
                  fitted-sources
                  fitted-weights
                  prediction->lists)
         (only-in "private/python-env.rkt" run-twin test-twin twin-params))

(define-runtime-path twin "python/hdbscan_twin.py")

(define defaults (hash-remove (hdbscan-params) 'resources))

(struct twin-case (name rows points options))

(define (fit-case name data options)
  (define rows (dataset data))
  (twin-case name (alternate rows 0) (alternate rows 1) options))

(define cuml-cases
  (list (fit-case "b50" 'b50 (hasheq))
        (fit-case "b10" 'b10 (hasheq))
        (fit-case "moons" 'moons (hasheq))
        (fit-case "circles" 'circles (hasheq))
        (fit-case "b5d4" 'b5d4 (hasheq 'min-cluster-size 10 'min-samples 3))
        (fit-case "b10, leaf" 'b10 (hasheq 'cluster-selection-method 'leaf))
        (fit-case "moons, max-cluster-size 50" 'moons (hasheq 'max-cluster-size 50))
        (fit-case "moons, alpha .5" 'moons (hasheq 'alpha 0.5))
        (fit-case "b10, epsilon 3" 'b10 (hasheq 'cluster-selection-epsilon 3.0))
        (fit-case "one-blob, allow-single" 'one-blob (hasheq 'allow-single-cluster #t))))

(define (sklearn-case name data options)
  (twin-case name (dataset data) '() options))

(define sklearn-cases
  (append (for/list ([data (in-list '(b50 b10 b5d4 moons gridclust))])
            (sklearn-case (symbol->string data) data (hasheq)))
          (list (sklearn-case "b10, epsilon 3" 'b10 (hasheq 'cluster-selection-epsilon 3.0))
                (sklearn-case "moons, alpha .5" 'moons (hasheq 'alpha 0.5))
                (sklearn-case "moons, allow-single at 600"
                              'moons
                              (hasheq 'allow-single-cluster #t 'min-cluster-size 600)))))

(define (options-of c)
  (for/fold ([options (hash-remove* defaults '(metric build-algo))])
            ([(k v) (in-hash (twin-case-options c))])
    (hash-set options k v)))

(define (python-case op c)
  (define o (options-of c))
  (hasheq 'op op
          'X (twin-case-rows c)
          'points (twin-case-points c)
          'min_cluster_size (hash-ref o 'min-cluster-size)
          'min_samples (or (hash-ref o 'min-samples) (json-null))
          'cluster_selection_epsilon (hash-ref o 'cluster-selection-epsilon)
          'max_cluster_size (hash-ref o 'max-cluster-size)
          'alpha (hash-ref o 'alpha)
          'cluster_selection_method (symbol->string (hash-ref o 'cluster-selection-method))
          'allow_single_cluster (hash-ref o 'allow-single-cluster)))

(define exact-fields
  (list (cons 'labels fitted-labels)
        (cons 'probabilities fitted-probabilities)
        (cons 'parents fitted-parents)
        (cons 'children fitted-children)
        (cons 'lambdas fitted-lambdas)
        (cons 'sizes fitted-sizes)
        (cons 'linkage fitted-linkage)
        (cons 'distances fitted-distances)
        (cons 'linkage_sizes fitted-linkage-sizes)
        (cons 'sources fitted-sources)
        (cons 'destinations fitted-destinations)
        (cons 'weights fitted-weights)))

(define (ours c)
  (define X (mat (twin-case-rows c) 'float32))
  (define points (mat (twin-case-points c) 'float32))
  (define result (apply-options hdbscan-fit (twin-case-options c) X))
  (values (fit-lists result)
          (hasheq 'predicted (prediction->lists (hdbscan-approximate-predict result points))
                  'all_points (device-matrix->list* (hdbscan-all-points-membership-vectors result))
                  'membership (device-matrix->list* (hdbscan-membership-vector result points)))))

(test-twin "T1, T2: HDBSCAN's trees and predictions equal cuML's; its labels are sklearn's, but on the Racket-drawn circles"
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin
              (hasheq 'cases (append (map (lambda (c) (python-case "fit" c)) cuml-cases)
                                     (map (lambda (c) (python-case "sklearn" c)) sklearn-cases)))))
  (check-equal? version "26.08.00")
  (check-equal? version libcuml-version)
  (define-values (cuml-results sklearn-results) (split-at results (length cuml-cases)))
  (define cuml-same
    (for/list ([c (in-list cuml-cases)]
               [theirs (in-list cuml-results)])
      (define-values (f predictions) (ours c))
      (define same
        (append (for/list ([field (in-list exact-fields)])
                  (match-define (cons name get) field)
                  (define equal-field? (equal? (get f) (hash-ref theirs name)))
                  (test-case (format "~a: ~a" (twin-case-name c) name)
                    (check-true equal-field?))
                  equal-field?)
                (for/list ([(key value) (in-hash predictions)])
                  (define equal-field? (equal? value (hash-ref theirs key)))
                  (test-case (format "~a: ~a" (twin-case-name c) key)
                    (check-true equal-field?))
                  equal-field?)))
      (test-case (format "~a: persistence" (twin-case-name c))
        (check-true (close? (fitted-persistence f) (hash-ref theirs 'persistence) 1e-5)))
      (andmap values same)))
  (define sklearn-same
    (for/list ([c (in-list sklearn-cases)]
               [theirs (in-list sklearn-results)])
      (define found
        (fitted-labels (fit-lists (apply-options hdbscan-fit
                                                 (twin-case-options c)
                                                 (mat (twin-case-rows c) 'float32)))))
      (test-case (format "sklearn: ~a" (twin-case-name c))
        (check-equal? found (hash-ref theirs 'labels)))
      (equal? found (hash-ref theirs 'labels))))
  (printf "twin summary: ~a cuML cases equal in every array in ~a; ~a sklearn cases equal in ~a\n"
          (length cuml-cases)
          (count values cuml-same)
          (length sklearn-cases)
          (count values sklearn-same)))

(test-twin "T3: cuml's HDBSCAN default parameters are hdbscan-params'"
  (match-define (hash-table ('results (list theirs)))
    (run-twin twin (hasheq 'cases (list (hasheq 'op "params")))))
  (define mapped (twin-params theirs '(p gen-min-span-tree prediction-data build-kwds device-ids)))
  (printf "twin params: ~s\n" theirs)
  (check-equal? mapped defaults)
  (check-equal? (hash-remove (estimator-params (hdbscan)) 'resources) defaults))
