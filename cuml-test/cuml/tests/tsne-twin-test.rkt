#lang racket/base

(require (only-in cuml/manifold tsne-params)
         (only-in racket/match match-define)
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" run-cases test-twin twin-params)
         (only-in "private/tsne-data.rkt" check-tsne-parity twin)
         (only-in "private/umap-data.rkt" digits held-out-fit))

(define held-out-configs
  (for*/list ([init (in-list '(random pca))]
              [metric (in-list '(euclidean sqeuclidean manhattan chebyshev cosine))])
    (hasheq 'init
            init
            'metric
            metric
            'perplexity
            20.0
            'early-exaggeration
            8.0
            'learning-rate
            150.0
            'max-iter
            400)))

(test-twin "L5: FFT fits equal Python cuML's bitwise on held-out blobs, every metric and init"
  (check-tsne-parity held-out-fit held-out-configs))

(test-twin "L5: FFT fits equal Python cuML's bitwise on digits at the defaults"
  (check-tsne-parity (digits) (list (hasheq 'init 'random) (hasheq 'init 'pca))))

(test-twin "L13: perplexity 30 takes 90 neighbours, 5 takes 15, and an explicit k wins"
  (check-tsne-parity held-out-fit
                     (list (hasheq 'perplexity 30.0)
                           (hasheq 'perplexity 5.0)
                           (hasheq 'perplexity 5.0 'n-neighbors 40))))

(define unbound
  '(late-exaggeration n-iter-without-progress
                      min-grad-norm
                      metric-params
                      angle
                      perplexity-max-iter
                      exaggeration-iter
                      pre-momentum
                      post-momentum
                      learning-rate-method
                      square-distances
                      precomputed-knn
                      handle))

(test-twin "L12: the parameters are Python's, but for random-state and n-neighbors"
  (match-define (list theirs) (run-cases twin (list (hasheq 'op "params"))))
  (check-equal? (twin-params theirs unbound)
                (hash-set* (tsne-params) 'random-state #f 'n-neighbors 90)))
