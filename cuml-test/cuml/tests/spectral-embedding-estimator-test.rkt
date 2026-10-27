#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  spectral-embedding
                  spectral-embedding-model
                  spectral-embedding?
                  transform)
         (only-in cuml/manifold
                  spectral-embedding-fit
                  spectral-embedding-fit-transform
                  spectral-embedding-params
                  spectral-embedding-result-embedding
                  spectral-embedding-result?)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" listed printed round-trips)
         (only-in "private/python-env.rkt" run-cases test-twin twin-params)
         (only-in "private/scaler-data.rkt" bits-equal?)
         (only-in "private/spectral-embedding-data.rkt" column-gaps embedded embedded-graph sbm twin)
         (only-in "private/umap-data.rkt" far-blobs fixture-X))

(define changed
  (hasheq 'n-components 3
          'n-neighbors 12
          'norm-laplacian #f
          'drop-first #f
          'random-state 7))

(test-case "L14: the default parameters, and every keyword round-trips"
  (check-equal? (spectral-embedding-params)
                (hasheq 'n-components 2
                        'n-neighbors #f
                        'norm-laplacian #t
                        'drop-first #t
                        'random-state 0
                        'affinity 'nearest-neighbors))
  (check-equal? (estimator-params (spectral-embedding)) (spectral-embedding-params))
  (check-equal? (round-trips spectral-embedding changed) changed))

(test-case "L14: printing, equality and set-params"
  (check-equal? (printed (spectral-embedding)) (make-list 3 "#<spectral-embedding>"))
  (check-equal? (format "~v" (spectral-embedding #:n-neighbors 10))
                "#<spectral-embedding #:n-neighbors 10>")
  (check-equal? (spectral-embedding #:drop-first #f) (spectral-embedding #:drop-first #f))
  (check-not-equal? (spectral-embedding #:n-components 3) (spectral-embedding))
  (check-equal? (estimator-set-params (spectral-embedding) #:norm-laplacian #f)
                (spectral-embedding #:norm-laplacian #f))
  (check-cuml-error 'logic
                    "spectral-embedding-params: spectral-embedding has no parameter gamma"
                    (lambda () (spectral-embedding-params (hasheq 'gamma 1.0))))
  (check-false (spectral-embedding? 'spectral-embedding))
  (check-false (spectral-embedding-model (spectral-embedding))))

(test-case "L14: spectral-embedding is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (spectral-embedding) spectral-embedding?) '(#t #f #f #t #f #t)))

(test-case "L14: transform is refused, fitted or not"
  (check-cuml-error 'logic
                    "transform: spectral-embedding has no transform; use fit-transform"
                    (lambda () (transform (spectral-embedding) 'X))))

(test-gpu "L14: fit-transform's matrix is the procedures', and transform is refused after a fit"
  (define X (fixture-X far-blobs))
  (define-values (result E) (spectral-embedding-fit-transform X #:random-state 3))
  (check-true (eq? E (spectral-embedding-result-embedding result)))
  (check-true (bits-equal? (listed E)
                           (listed (spectral-embedding-result-embedding
                                    (spectral-embedding-fit X #:random-state 3)))))
  (check-true (bits-equal? (listed (fit-transform (spectral-embedding #:random-state 3) X))
                           (listed E)))
  (define e (fit (spectral-embedding) X))
  (check-equal? (list (fitted? (spectral-embedding)) (fitted? e)) '(#f #t))
  (check-equal? (format "~v" e) "#<spectral-embedding fitted>")
  (check-true (spectral-embedding-result? (spectral-embedding-model e)))
  (check-true (bits-equal? (listed (spectral-embedding-result-embedding (spectral-embedding-model e)))
                           (listed (spectral-embedding-result-embedding
                                    (spectral-embedding-fit X)))))
  (check-cuml-error 'logic
                    "transform: spectral-embedding has no transform; use fit-transform"
                    (lambda () (transform e X))))

(test-gpu "L14: (random-seed 7) repeats a drawn seed; seeds 0 and 1 differ in bits, not in the bound"
  (define-values (g _truth) (sbm '(40 60 80) 0.3 0.01 0))
  (define (drawn)
    (random-seed 7)
    (embedded (embedded-graph g (hasheq 'random-state #f))))
  (check-true (bits-equal? (drawn) (drawn)))
  (define zero (embedded (embedded-graph g (hasheq 'random-state 0))))
  (define one (embedded (embedded-graph g (hasheq 'random-state 1))))
  (check-false (bits-equal? zero one))
  (check-true (<= (apply max (column-gaps one zero)) 1e-4) (format "~a" (column-gaps one zero))))

(test-twin "L14: the parameters are Python's, but for random-state 0 and the function-only keywords"
  (match-define (list theirs) (run-cases twin (list (hasheq 'op "params"))))
  (check-equal? (hash-set (twin-params theirs '()) 'random-state 0)
                (hash-remove (hash-remove (spectral-embedding-params) 'norm-laplacian) 'drop-first)
                (format "~a" theirs)))
