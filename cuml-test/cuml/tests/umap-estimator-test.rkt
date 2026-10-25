#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  transform
                  umap
                  umap-model
                  umap?)
         (only-in cuml/manifold
                  umap-fit
                  umap-fit-transform
                  umap-params
                  umap-result-embedding
                  umap-result?
                  umap-transform)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" listed printed round-trips)
         (only-in "private/umap-data.rkt" bits-equal? far-blobs fixture-X fixture-y overlap))

(define changed
  (hasheq 'n-neighbors
          5
          'n-components
          3
          'metric
          'cosine
          'n-epochs
          50
          'learning-rate
          0.5
          'min-dist
          0.2
          'spread
          2.0
          'a
          1.5
          'b
          0.9
          'init
          'random
          'negative-sample-rate
          3
          'random-state
          7
          'build-algo
          'nn-descent))

(test-case "L15: the default parameters, and every keyword round-trips"
  (check-equal? (umap-params)
                (hasheq 'n-neighbors 15
                        'n-components 2
                        'metric 'euclidean
                        'n-epochs #f
                        'learning-rate 1.0
                        'min-dist 0.1
                        'spread 1.0
                        'a #f
                        'b #f
                        'init 'spectral
                        'negative-sample-rate 5
                        'random-state 0
                        'build-algo 'brute-force))
  (check-equal? (estimator-params (umap)) (umap-params))
  (check-equal? (round-trips umap changed) changed))

(test-case "L15: printing, equality and set-params"
  (check-equal? (printed (umap)) (make-list 3 "#<umap>"))
  (check-equal? (format "~v" (umap #:n-neighbors 5)) "#<umap #:n-neighbors 5>")
  (check-equal? (umap #:init 'random) (umap #:init 'random))
  (check-not-equal? (umap #:spread 2.0) (umap))
  (check-equal? (equal-hash-code (umap #:min-dist 0.2)) (equal-hash-code (umap #:min-dist 0.2)))
  (check-equal? (estimator-set-params (umap) #:n-components 4) (umap #:n-components 4))
  (check-cuml-error 'logic
                    "umap-params: umap has no parameter local-connectivity"
                    (lambda () (umap-params (hasheq 'local-connectivity 2.0))))
  (check-false (umap? 'umap))
  (check-false (umap-model (umap))))

(test-case "L15: umap is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (umap) umap?) '(#t #f #f #t #f #t)))

(test-gpu "L11: fit-transform's matrix is the fit's embedding, and the estimator is the procedures"
  (define X (fixture-X far-blobs))
  (define-values (result E) (umap-fit-transform X))
  (check-true (eq? E (umap-result-embedding result)))
  (check-true (bits-equal? (listed (fit-transform (umap) X)) (listed E)))
  (define e (fit (umap) X))
  (check-equal? (list (fitted? (umap)) (fitted? e)) '(#f #t))
  (check-equal? (format "~v" e) "#<umap fitted>")
  (check-equal? (format "~v" (fit (umap #:n-neighbors 5) X)) "#<umap fitted #:n-neighbors 5>")
  (check-true (umap-result? (umap-model e)))
  (check-true (bits-equal? (listed (umap-result-embedding (umap-model e))) (listed E)))
  (check-true (bits-equal? (listed (transform e X)) (listed (umap-transform (umap-model e) X)))))

(test-gpu "L11: the generic fit passes y as #:y"
  (define X (fixture-X overlap))
  (define y (fixture-y overlap))
  (define supervised (listed (umap-result-embedding (umap-fit X #:y y #:init 'random))))
  (check-true (bits-equal? (listed (umap-result-embedding (umap-model (fit (umap #:init 'random) X y))))
                           supervised))
  (check-true (bits-equal? (listed (fit-transform (umap #:init 'random) X y)) supervised))
  (check-false (bits-equal? (listed (fit-transform (umap #:init 'random) X)) supervised)))
