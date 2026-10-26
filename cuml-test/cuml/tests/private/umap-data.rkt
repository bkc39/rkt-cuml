#lang racket/base

(require (only-in cuml/manifold
                  umap-fit
                  umap-result-a
                  umap-result-b
                  umap-result-embedding
                  umap-result-params
                  umap-transform)
         (only-in cuml/metrics trustworthiness)
         (only-in json json-null)
         (only-in racket/list drop last range take)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list* list*->device-matrix list->device-vector)
         (only-in "blob-data.rkt" centred-blobs)
         (only-in "metrics-data.rkt" median)
         (only-in "python-env.rkt" run-twin)
         (only-in "scaler-data.rkt" bits-equal?))

(provide (struct-out fixture)
         blobs
         case-of
         check-parity
         check-quality
         digits
         embedding-rows
         far-blobs
         finite-rows?
         fixture-X
         fixture-y
         held-out-fit
         held-out-new
         jittered
         overlap
         ours
         relabel
         tail-split
         tiny
         twin-case)

(struct fixture (name rows labels))

(define-runtime-path twin "../python/umap_twin.py")

(define (grouped name seed centres per-group)
  (define-values (rows labels) (centred-blobs centres per-group seed))
  (fixture name rows labels))

(define (axis d i value)
  (for/list ([j (in-range d)])
    (if (= i j) value 0.0)))

(define blobs
  (grouped "blobs"
           51
           (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
             (random-seed 50)
             (for/list ([_ (in-range 5)])
               (for/list ([_ (in-range 10)])
                 (+ -10.0 (* 20.0 (random))))))
           200))

(define (every-fifth? i)
  (= (modulo i 5) 4))

(define (split-fixture f)
  (define-values (kept-rows kept-labels new-rows new-labels)
    (for/fold ([kept-rows '()]
               [kept-labels '()]
               [new-rows '()]
               [new-labels '()]
               #:result (values (reverse kept-rows)
                                (reverse kept-labels)
                                (reverse new-rows)
                                (reverse new-labels)))
              ([row (in-list (fixture-rows f))]
               [label (in-list (fixture-labels f))]
               [i (in-naturals)])
      (if (every-fifth? i)
          (values kept-rows kept-labels (cons row new-rows) (cons label new-labels))
          (values (cons row kept-rows) (cons label kept-labels) new-rows new-labels))))
  (values (fixture (fixture-name f) kept-rows kept-labels)
          (fixture (fixture-name f) new-rows new-labels)))

(define-values (held-out-fit held-out-new) (split-fixture blobs))

(define far-blobs (grouped "far-blobs" 52 (list (axis 5 0 0.0) (axis 5 0 50.0) (axis 5 1 50.0)) 100))

(define overlap (grouped "overlap" 53 (list (axis 10 0 0.0) (axis 10 0 3.0)) 300))

(define tiny
  (fixture "tiny"
           '((0.125 0.0 0.0) (0.5 0.25 0.0)
                             (1.0 0.0 0.5)
                             (0.25 1.0 0.0)
                             (10.0 10.0 10.0)
                             (10.5 10.25 10.0)
                             (11.0 10.0 10.5)
                             (10.25 11.0 10.0))
           '(0 0 0 0 1 1 1 1)))

(define (fixture-X f)
  (list*->device-matrix (fixture-rows f) #:dtype 'float32))

(define (fixture-y f [dtype 'int32])
  (list->device-vector (fixture-labels f) #:dtype dtype))

(define (relabel labels)
  (for/list ([y (in-list labels)])
    (if (= y -1)
        -1
        (+ (* (expt 2 40) y) 7))))

(define (embedding-rows E)
  (device-matrix->list* E))

(define (finite-rows? rows)
  (for*/and ([row (in-list rows)]
             [x (in-list row)])
    (and (flonum? x) (< (abs x) +inf.0))))

(define (twin-case f result-params a b #:transform [Y #f] #:y [y #f] #:op [op "umap"])
  (define (given key)
    (hash-ref result-params key))
  (hasheq 'op op
          'X (fixture-rows f)
          'y (or y (json-null))
          'Y (or Y (json-null))
          'n_neighbors (given 'n-neighbors)
          'n_components (given 'n-components)
          'metric (symbol->string (given 'metric))
          'n_epochs (or (given 'n-epochs) (json-null))
          'learning_rate (given 'learning-rate)
          'min_dist (given 'min-dist)
          'spread (given 'spread)
          'a a
          'b b
          'init (symbol->string (given 'init))
          'negative_sample_rate (given 'negative-sample-rate)
          'seed (given 'random-state)
          'build_algo (if (eq? (given 'build-algo) 'nn-descent) "nn_descent" "brute_force_knn")))

(define (digits)
  (match-define (list (hash-table ['X rows] ['y labels]))
    (hash-ref (run-twin twin (hasheq 'cases (list (hasheq 'op "digits")))) 'results))
  (fixture "digits" rows labels))

(define (tail-split f)
  (match-define (fixture name rows labels) f)
  (define cut (quotient (* 4 (length rows)) 5))
  (values (fixture name (take rows cut) (take labels cut))
          (fixture (format "~a tail" name) (drop rows cut) (drop labels cut))))

(define (jittered f seed)
  (match-define (fixture name rows labels) f)
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed seed)
    (fixture (format "~a jittered" name)
             (for/list ([row (in-list rows)])
               (for/list ([x (in-list row)])
                 (+ x (* 0.01 (random)))))
             labels)))

(define (check-within name ours spread)
  (define floor (apply min spread))
  (define delta (- (apply max spread) floor))
  (define centre (median spread))
  (printf "~a: ours ~a, Python min ~a median ~a delta ~a\n" name ours floor centre delta)
  (check-true (>= ours (- floor delta)) (format "~a: ~a below ~a - ~a" name ours floor delta))
  (check-true (<= (abs (- ours centre)) delta) (format "~a: ~a off the median ~a" name ours centre)))

(define (quality-case f result #:transform [Y #f])
  (hash-set* (twin-case f
                        (umap-result-params result)
                        (umap-result-a result)
                        (umap-result-b result)
                        #:transform Y
                        #:op "quality")
             'seeds
             (range 20)))

(define (check-quality init fits held new)
  (define results
    (for/list ([f (in-list fits)])
      (umap-fit (fixture-X f) #:init init)))
  (define held-result (umap-fit (fixture-X held) #:init init))
  (define found
    (hash-ref (run-twin twin
                        (hasheq 'cases (append (map quality-case fits results)
                                               (list (quality-case held
                                                                   held-result
                                                                   #:transform (fixture-rows new))))))
              'results))
  (for ([f (in-list fits)]
        [result (in-list results)]
        [theirs (in-list found)])
    (test-case (format "~a fit ~a" init (fixture-name f))
      (check-within (format "~a fit ~a" init (fixture-name f))
                    (trustworthiness (fixture-X f) (umap-result-embedding result) #:n-neighbors 10)
                    (hash-ref theirs 'trustworthiness))))
  (test-case (format "~a transform ~a" init (fixture-name new))
    (check-within
     (format "~a transform ~a" init (fixture-name new))
     (trustworthiness (fixture-X new) (umap-transform held-result (fixture-X new)) #:n-neighbors 10)
     (hash-ref (last found) 'trustworthiness))))

(define (ours f new options #:y [y #f])
  (define result
    (keyword-apply umap-fit (map car options) (map cdr options) (list (fixture-X f)) #:y y))
  (list result
        (device-matrix->list* (umap-result-embedding result))
        (device-matrix->list* (umap-transform result (fixture-X new)))))

(define (case-of f new result #:y [y #f])
  (twin-case f
             (umap-result-params result)
             (umap-result-a result)
             (umap-result-b result)
             #:y y
             #:transform (fixture-rows new)))

(define (check-parity f new configs #:transform? [transform? #t])
  (define runs
    (for/list ([options (in-list configs)])
      (ours f new options)))
  (define theirs
    (hash-ref (run-twin twin
                        (hasheq 'cases (for/list ([run (in-list runs)])
                                         (case-of f new (car run)))))
              'results))
  (for ([options (in-list configs)]
        [run (in-list runs)]
        [found (in-list theirs)])
    (match-define (list _ embedding transformed) run)
    (test-case (format "~a ~a" (fixture-name f) options)
      (check-true (bits-equal? embedding (hash-ref found 'embedding)) "fit")
      (when transform?
        (check-true (bits-equal? transformed (hash-ref found 'transform)) "transform")))))
