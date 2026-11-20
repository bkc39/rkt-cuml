#lang racket/base

(require (only-in cuml/decomposition pca-params)
         (only-in racket/match match-define)
         (only-in rackunit check-equal?)
         (only-in "private/pca-twin-data.rkt" check-against-cuml named twin-results)
         (only-in "private/python-env.rkt" test-twin twin-params))

(test-twin "normal is cuML-Python's within the L14 bounds"
  (check-against-cuml (named "normal")))

(define renames (hasheq 'iterated-power 'iterations 'svd-solver 'solver))

(test-twin "the hyperparameters are cuML-Python's but copy, with two departures"
  (match-define (list theirs) (twin-results (list (hasheq 'op "params"))))
  (define mapped
    (for/hasheq ([(key value) (in-hash (twin-params theirs '(copy)))])
      (values (hash-ref renames key key) value)))
  (check-equal? (hash-ref mapped 'tol) 1e-7)
  (check-equal? (hash-ref mapped 'solver) 'auto)
  (check-equal? (hash-set* mapped 'tol 0.0 'solver 'full) (pca-params)))
