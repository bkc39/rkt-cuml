#lang racket/base

(require (only-in cuml fit pca transform)
         (only-in cuml/decomposition
                  pca-fit
                  pca-inverse-transform
                  pca-params
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-noise-variance
                  pca-result-singular-values
                  pca-transform)
         (only-in rackunit check-equal? check-true)
         (only-in raft device-matrix->list* device-vector->list list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define axes '((4 0) (-4 0) (0 1) (0 -1) (0 0)))
(define exact-axes '((4.0 0.0) (-4.0 0.0) (0.0 1.0) (0.0 -1.0) (0.0 0.0)))

(define (X)
  (list*->device-matrix axes #:dtype 'float64))

(define (same? got want)
  (and (= (length got) (length want))
       (for/and ([g (in-list got)]
                 [w (in-list want)])
         (if (list? w) (same? g w) (= g w)))))

(test-gpu "the manual: the axis example's components, spectrum and transform"
  (define fitted (pca-fit (X)))
  (check-equal? (device-matrix->list* (pca-result-components fitted)) '((1.0 0.0) (0.0 1.0)))
  (check-equal? (map device-vector->list
                     (list (pca-result-explained-variance fitted)
                           (pca-result-explained-variance-ratio fitted)
                           (pca-result-singular-values fitted)))
                (list '(8.0 0.5) (list (/ 8.0 8.5) (/ 0.5 8.5)) (list (sqrt 32.0) (sqrt 2.0))))
  (check-true (same? (device-matrix->list* (pca-transform fitted (X))) exact-axes))
  (check-true (same? (device-matrix->list* (pca-inverse-transform fitted (pca-transform fitted (X))))
                     exact-axes)))

(test-gpu "the manual: noise variance and the fit's refusals"
  (check-equal? (pca-result-noise-variance (pca-fit (X) #:n-components 1)) 0.5)
  (check-cuml-error 'logic
                    "pca-fit: n-components: expected at most 2, got 3"
                    (lambda () (pca-fit (X) #:n-components 3)))
  (check-cuml-error 'logic
                    "pca-fit: X: expected at least 2 distinct rows, got 1"
                    (lambda () (pca-fit (list*->device-matrix '((1 2) (1 2)) #:dtype 'float64)))))

(test-gpu "the manual: whitening, and a transform refusal"
  (define whitened (device-matrix->list* (pca-transform (pca-fit (X) #:whiten #t) (X))))
  (check-true (same? whitened
                     (for/list ([row (in-list exact-axes)])
                       (map / row (list (sqrt 8.0) (sqrt 0.5))))))
  (check-cuml-error
   'logic
   "pca-transform: X: expected 2 columns, got 3"
   (lambda () (pca-transform (pca-fit (X)) (list*->device-matrix '((1 2 3)) #:dtype 'float64)))))

(test-gpu "the manual: pca-params and the estimator"
  (check-equal? (hash-ref (pca-params #:solver 'jacobi) 'solver) 'jacobi)
  (check-cuml-error 'logic
                    "pca-params: pca has no parameter copy"
                    (lambda () (pca-params (hasheq 'copy #t))))
  (check-true (same? (device-matrix->list* (transform (fit (pca #:n-components 1) (X)) (X)))
                     '((4.0) (-4.0) (0.0) (0.0) (0.0))))
  (check-cuml-error 'logic "transform: pca is not fitted" (lambda () (transform (pca) (X)))))
