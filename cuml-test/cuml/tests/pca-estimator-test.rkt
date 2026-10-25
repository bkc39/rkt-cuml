#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  inverse-transform
                  pca
                  pca-model
                  pca?
                  transform)
         (only-in cuml/decomposition
                  pca-fit
                  pca-fit-transform
                  pca-inverse-transform
                  pca-params
                  pca-result-n-samples
                  pca-result?
                  pca-transform)
         (only-in racket/list make-list take)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" round-trips)
         (only-in "private/pca-data.rkt" fitted-of fixture-rows integer-fixture normal-fixture)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix))

(define changed (hasheq 'n-components 2 'whiten #t 'solver 'jacobi 'tol 1e-6 'iterations 30))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "L18: the default parameters, and every keyword round-trips"
  (check-equal? (pca-params) (hasheq 'n-components #f 'whiten #f 'solver 'full 'tol 0.0 'iterations 15))
  (check-equal? (estimator-params (pca)) (pca-params))
  (check-equal? (round-trips pca changed) changed)
  (check-equal? (estimator-params (pca (pca-params #:whiten #t) #:n-components 3))
                (pca-params #:whiten #t #:n-components 3)))

(test-case "L18: printing, equality and set-params"
  (check-equal? (printed (pca)) (make-list 3 "#<pca>"))
  (check-equal? (format "~v" (pca #:n-components 2)) "#<pca #:n-components 2>")
  (check-equal? (pca #:solver 'jacobi) (pca #:solver 'jacobi))
  (check-not-equal? (pca #:whiten #t) (pca))
  (check-equal? (equal-hash-code (pca #:tol 1e-3)) (equal-hash-code (pca #:tol 1e-3)))
  (check-equal? (estimator-set-params (pca) #:n-components 4) (pca #:n-components 4))
  (check-cuml-error 'logic
                    "pca-params: pca has no parameter copy"
                    (lambda () (pca-params (hasheq 'copy #t))))
  (check-false (pca? 'pca))
  (check-false (pca-model (pca))))

(test-case "L18: pca is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (pca) pca?) '(#t #f #f #t #f #t)))

(define (listed M)
  (device-matrix->list* M))

(test-gpu "L13, L18: the estimator is the procedures, bitwise, on the same fit"
  (for* ([dt (in-list '(float32 float64))]
         [solver (in-list '(full jacobi))]
         [whiten (in-list '(#f #t))])
    (test-case (format "~a ~a ~a" dt solver whiten)
      (define X (rows->matrix (take (fixture-rows normal-fixture dt) 300) dt))
      (define e (pca #:n-components 3 #:solver solver #:whiten whiten))
      (define fitted-e (fit e X '(ignored)))
      (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
      (check-equal? (format "~v" fitted-e) "#<pca fitted>")
      (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
      (define result (pca-model fitted-e))
      (check-true (pca-result? result))
      (check-equal? (pca-result-n-samples result) 300)
      (define T (pca-transform result X))
      (check-true (bits-equal? (listed (transform fitted-e X)) (listed T)))
      (check-true (bits-equal? (listed (inverse-transform fitted-e T))
                               (listed (pca-inverse-transform result T)))))))

(test-gpu "L13b, L18: on exact sums the fits agree bitwise, so fit-transform is fit then transform"
  (for ([dt (in-list '(float32 float64))])
    (define X (rows->matrix (fixture-rows integer-fixture dt) dt))
    (define result (pca-fit X #:n-components 5))
    (define T (listed (pca-transform result X)))
    (define-values (again T2) (pca-fit-transform X #:n-components 5))
    (check-equal? (fitted-of again) (fitted-of result))
    (check-true (bits-equal? (listed T2) T))
    (check-true (bits-equal? (listed (fit-transform (pca #:n-components 5) X)) T))))

(test-gpu "L16: fit-transform names the generic for a fit refusal"
  (check-cuml-error 'logic
                    "fit-transform: X: expected at least 2 rows, got 1"
                    (lambda () (fit-transform (pca) (rows->matrix '((1 2 3)) 'float32)))))
