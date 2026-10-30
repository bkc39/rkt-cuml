#lang racket/base

(require (only-in cuml fit fit-transform spectral-embedding transform tsne umap)
         (only-in cuml/cluster hdbscan-fit hdbscan-result-labels kmeans-fit kmeans-result-labels)
         (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-graph
                  spectral-embedding-params
                  spectral-embedding-result-embedding
                  spectral-embedding-result-n-neighbors
                  tsne-fit
                  tsne-params
                  tsne-result-embedding
                  tsne-result-n-iter
                  umap-fit
                  umap-params
                  umap-result-a
                  umap-result-embedding
                  umap-transform)
         (only-in cuml/manifold/private/curve curve-parameters)
         (only-in cuml/metrics adjusted-rand-score)
         (only-in racket/flonum flsingle)
         (only-in racket/match match-define)
         (only-in rackunit check-equal?)
         (only-in raft list*->device-matrix list->device-vector shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/umap-data.rkt" fixture-X held-out-fit))

(define (X)
  (list*->device-matrix (for*/list ([c (in-list '(0 25 50))]
                                    [i (in-range 20)])
                          (list (+ c (* 0.1 i)) (- c (* 0.05 i)) (* 0.01 i i)))
                        #:dtype 'float32))

(define truth
  (for*/list ([k 3]
              [_ 20])
    k))

(test-gpu "the manual: the three groups example"
  (define fitted (umap-fit (X) #:n-neighbors 10))
  (check-equal? (shape (umap-result-embedding fitted)) '(60 2))
  (check-equal? (~> fitted
                    umap-result-embedding
                    (hdbscan-fit #:min-cluster-size 10)
                    hdbscan-result-labels
                    (adjusted-rand-score (list->device-vector truth #:dtype 'int32) _))
                1.0)
  (check-equal? (shape (umap-transform fitted (X))) '(60 2)))

(test-gpu "the manual: the curve parameter and the fit's refusal"
  (define-values (a _b) (curve-parameters 0.1 1.0))
  (check-equal? (umap-result-a (umap-fit (X) #:n-neighbors 10)) (flsingle a))
  (check-cuml-error 'logic
                    "umap-fit: min-dist: expected at most the spread 1.0, got 2.0"
                    (lambda () (umap-fit (X) #:min-dist 2.0))))

(test-gpu "the manual: umap-params and the estimator"
  (check-equal? (hash-ref (umap-params #:init 'random) 'init) 'random)
  (check-cuml-error 'logic
                    "umap-params: umap has no parameter local-connectivity"
                    (lambda () (umap-params (hasheq 'local-connectivity 2.0))))
  (check-equal? (shape (fit-transform (umap #:n-neighbors 10) (X))) '(60 2))
  (check-cuml-error 'logic "transform: umap is not fitted" (lambda () (transform (umap) (X)))))

(test-gpu "the manual: spectral needs at least 2(n-components + 2) rows"
  (check-cuml-error
   'logic
   "umap-fit: n-components: spectral init of 399 components needs at least 802 rows, got 800"
   (lambda () (umap-fit (fixture-X held-out-fit) #:n-components 399))))

(test-gpu "the manual: the t-SNE example, tsne-fit and tsne-params"
  (define embedded (tsne-fit (X) #:perplexity 5))
  (check-equal? (shape (tsne-result-embedding embedded)) '(60 2))
  (check-equal? (~> embedded
                    tsne-result-embedding
                    (hdbscan-fit #:min-cluster-size 10)
                    hdbscan-result-labels
                    (adjusted-rand-score (list->device-vector truth #:dtype 'int32) _))
                1.0)
  (check-equal? (tsne-result-n-iter embedded) 1000)
  (check-cuml-error 'logic
                    "tsne-fit: perplexity: expected at most the 60 neighbours, got 100"
                    (lambda () (tsne-fit (X) #:perplexity 100)))
  (check-equal? (hash-ref (tsne-params #:method 'exact) 'method) 'exact)
  (check-cuml-error 'logic
                    "tsne-params: tsne has no parameter angle"
                    (lambda () (tsne-params (hasheq 'angle 0.5)))))

(test-gpu "the manual: the t-SNE estimator"
  (check-equal? (shape (fit-transform (tsne #:perplexity 5) (X))) '(60 2))
  (check-cuml-error 'logic
                    "transform: tsne has no transform; use fit-transform"
                    (lambda () (transform (fit (tsne #:perplexity 5) (X)) (X)))))

(define path-n 24)

(define (path-graph)
  (define path
    (for*/list ([i (sub1 path-n)]
                [e (list (cons i (add1 i)) (cons (add1 i) i))])
      e))
  (list (list->device-vector (map car path) #:dtype 'int32)
        (list->device-vector (map cdr path) #:dtype 'int32)
        (list->device-vector (map (lambda (_) 1.0) path) #:dtype 'float32)))

(test-gpu "the manual: the spectral embedding example, its fits and params"
  (define spectral (spectral-embedding-fit (X) #:n-neighbors 10))
  (check-equal? (shape (spectral-embedding-result-embedding spectral)) '(60 2))
  (check-equal? (~> spectral
                    spectral-embedding-result-embedding
                    (kmeans-fit #:n-clusters 3)
                    kmeans-result-labels
                    (adjusted-rand-score (list->device-vector truth #:dtype 'int32) _))
                1.0)
  (check-equal? (spectral-embedding-result-n-neighbors spectral) 10)
  (check-cuml-error
   'logic
   "spectral-embedding-fit: n-components: 30 components with drop-first need at least 64 rows, got 60"
   (lambda () (spectral-embedding-fit (X) #:n-components 30)))
  (match-define (list rows cols vals) (path-graph))
  (check-equal? (~> (spectral-embedding-fit-graph rows cols vals path-n #:n-components 1)
                    spectral-embedding-result-embedding
                    shape)
                '(24 1))
  (check-cuml-error
   'logic
   "spectral-embedding-fit-graph: rows: entry 0: self-loop at sample 0; leave the diagonal out"
   (lambda () (spectral-embedding-fit-graph rows rows vals path-n)))
  (check-equal? (hash-ref (spectral-embedding-params #:drop-first #f) 'drop-first) #f)
  (check-cuml-error
   'logic
   "spectral-embedding-fit: affinity: expected nearest-neighbors, got precomputed"
   (lambda ()
     (spectral-embedding-fit (X) #:params (spectral-embedding-params #:affinity 'precomputed)))))

(test-gpu "the manual: the spectral embedding estimator"
  (check-equal? (shape (fit-transform (spectral-embedding #:n-neighbors 10) (X))) '(60 2))
  (check-cuml-error 'logic
                    "transform: spectral-embedding has no transform; use fit-transform"
                    (lambda () (transform (fit (spectral-embedding #:n-neighbors 10) (X)) (X)))))
