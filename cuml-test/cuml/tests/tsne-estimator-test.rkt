#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  transform
                  tsne
                  tsne-model
                  tsne?)
         (only-in cuml/manifold
                  tsne-fit
                  tsne-fit-transform
                  tsne-params
                  tsne-result-embedding
                  tsne-result?)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" listed printed round-trips)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/umap-data.rkt" far-blobs fixture-X))

(define changed
  (hasheq 'n-components 2
          'perplexity 5.0
          'early-exaggeration 8.0
          'learning-rate 150.0
          'max-iter 300
          'n-neighbors 20
          'method 'exact
          'metric 'cosine
          'init 'pca
          'random-state 7))

(test-case "L12: the default parameters, and every keyword round-trips"
  (check-equal? (tsne-params)
                (hasheq 'n-components 2
                        'perplexity 30.0
                        'early-exaggeration 12.0
                        'learning-rate 200.0
                        'max-iter 1000
                        'n-neighbors #f
                        'method 'fft
                        'metric 'euclidean
                        'init 'random
                        'random-state 0))
  (check-equal? (estimator-params (tsne)) (tsne-params))
  (check-equal? (round-trips tsne changed) changed))

(test-case "L12: printing, equality and set-params"
  (check-equal? (printed (tsne)) (make-list 3 "#<tsne>"))
  (check-equal? (format "~v" (tsne #:perplexity 5.0)) "#<tsne #:perplexity 5.0>")
  (check-equal? (tsne #:init 'pca) (tsne #:init 'pca))
  (check-not-equal? (tsne #:max-iter 300) (tsne))
  (check-equal? (equal-hash-code (tsne #:method 'exact)) (equal-hash-code (tsne #:method 'exact)))
  (check-equal? (estimator-set-params (tsne) #:learning-rate 100.0) (tsne #:learning-rate 100.0))
  (check-cuml-error 'logic
                    "tsne-params: tsne has no parameter angle"
                    (lambda () (tsne-params (hasheq 'angle 0.5))))
  (check-false (tsne? 'tsne))
  (check-false (tsne-model (tsne))))

(test-case "L12: tsne is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (tsne) tsne?) '(#t #f #f #t #f #t)))

(test-case "L10: transform is refused, fitted or not"
  (check-cuml-error 'logic
                    "transform: tsne has no transform; use fit-transform"
                    (lambda () (transform (tsne) 'X))))

(test-gpu "L10: fit-transform's matrix is the fit's embedding, and the estimator is the procedures"
  (define X (fixture-X far-blobs))
  (define-values (result E) (tsne-fit-transform X #:random-state 3))
  (check-true (eq? E (tsne-result-embedding result)))
  (check-true (bits-equal? (listed E) (listed (tsne-result-embedding (tsne-fit X #:random-state 3)))))
  (check-true (bits-equal? (listed (fit-transform (tsne #:random-state 3) X)) (listed E)))
  (define e (fit (tsne) X))
  (check-equal? (list (fitted? (tsne)) (fitted? e)) '(#f #t))
  (check-equal? (format "~v" e) "#<tsne fitted>")
  (check-true (tsne-result? (tsne-model e)))
  (check-true (bits-equal? (listed (tsne-result-embedding (tsne-model e)))
                           (listed (tsne-result-embedding (tsne-fit X)))))
  (check-cuml-error 'logic
                    "transform: tsne has no transform; use fit-transform"
                    (lambda () (transform e X))))
