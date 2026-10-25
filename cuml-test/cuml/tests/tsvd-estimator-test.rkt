#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  inverse-transform
                  transform
                  truncated-svd
                  truncated-svd-model
                  truncated-svd?)
         (only-in cuml/decomposition
                  tsvd-fit
                  tsvd-fit-transform
                  tsvd-inverse-transform
                  tsvd-params
                  tsvd-result?
                  tsvd-transform)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" round-trips)
         (only-in "private/pca-data.rkt" fixture-rows)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix)
         (only-in "private/tsvd-data.rkt" normal-fixture offset-fixture tsvd-of))

(define changed (hasheq 'n-components 2 'solver 'jacobi 'tol 1e-6 'iterations 30))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(define (listed M)
  (device-matrix->list* M))

(test-case "L20: the default parameters, and every keyword round-trips"
  (check-equal? (tsvd-params) (hasheq 'n-components 1 'solver 'full 'tol 0.0 'iterations 15))
  (check-equal? (estimator-params (truncated-svd)) (tsvd-params))
  (check-equal? (round-trips truncated-svd changed) changed)
  (check-equal? (estimator-params (truncated-svd (tsvd-params #:solver 'jacobi) #:n-components 3))
                (tsvd-params #:solver 'jacobi #:n-components 3)))

(test-case "L20: printing, equality and set-params"
  (check-equal? (printed (truncated-svd)) (make-list 3 "#<truncated-svd>"))
  (check-equal? (format "~v" (truncated-svd #:n-components 2)) "#<truncated-svd #:n-components 2>")
  (check-equal? (truncated-svd #:solver 'jacobi) (truncated-svd #:solver 'jacobi))
  (check-not-equal? (truncated-svd #:n-components 2) (truncated-svd))
  (check-equal? (equal-hash-code (truncated-svd #:tol 1e-3)) (equal-hash-code (truncated-svd #:tol 1e-3)))
  (check-equal? (estimator-set-params (truncated-svd) #:n-components 4) (truncated-svd #:n-components 4))
  (check-cuml-error 'logic
                    "tsvd-params: tsvd has no parameter algorithm"
                    (lambda () (tsvd-params (hasheq 'algorithm 'full))))
  (check-false (truncated-svd? 'truncated-svd))
  (check-false (truncated-svd-model (truncated-svd))))

(test-case "L20: truncated-svd is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (truncated-svd) truncated-svd?) '(#t #f #f #t #f #t)))

(test-gpu "L13: fit-transform is fit then transform, bitwise, as procedures and as generics"
  (for* ([f (in-list (list normal-fixture offset-fixture))]
         [dt (in-list '(float32 float64))]
         [solver (in-list '(full jacobi))])
    (test-case (format "~a ~a" dt solver)
      (define X (rows->matrix (fixture-rows f dt) dt))
      (define result (tsvd-fit X #:n-components 3 #:solver solver))
      (define T (listed (tsvd-transform result X)))
      (define-values (again T2) (tsvd-fit-transform X #:n-components 3 #:solver solver))
      (check-equal? (tsvd-of again) (tsvd-of result))
      (check-true (bits-equal? (listed T2) T) "the procedure")
      (define e (truncated-svd #:n-components 3 #:solver solver))
      (check-true (bits-equal? (listed (fit-transform e X)) T) "the generic")
      (define fitted-e (fit e X '(ignored)))
      (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
      (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
      (check-true (tsvd-result? (truncated-svd-model fitted-e)))
      (check-equal? (tsvd-of (truncated-svd-model fitted-e)) (tsvd-of result))
      (check-true (bits-equal? (listed (transform fitted-e X)) T) "transform")
      (check-true (bits-equal? (listed (inverse-transform fitted-e (tsvd-transform result X)))
                               (listed (tsvd-inverse-transform result (tsvd-transform result X))))
                  "inverse-transform"))))

(test-gpu "L20: a fitted truncated-svd prints its parameters after fitted"
  (define X (rows->matrix (fixture-rows normal-fixture 'float32) 'float32))
  (check-equal? (format "~v" (fit (truncated-svd) X)) "#<truncated-svd fitted>")
  (check-equal? (format "~v" (fit (truncated-svd #:n-components 2) X))
                "#<truncated-svd fitted #:n-components 2>"))
