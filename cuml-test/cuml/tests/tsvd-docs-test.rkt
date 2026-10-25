#lang racket/base

(require (only-in cuml fit transform truncated-svd)
         (only-in cuml/decomposition
                  tsvd-fit
                  tsvd-inverse-transform
                  tsvd-params
                  tsvd-result-components
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values
                  tsvd-transform)
         (only-in rackunit check-equal? check-true)
         (only-in raft device-matrix->list* device-vector->list list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define rows '((1 2 0) (2 4 1) (3 6 0) (4 8 1)))

(define (X)
  (list*->device-matrix rows #:dtype 'float64))

(define (close? got want)
  (cond
    [(list? want) (and (= (length got) (length want)) (andmap close? got want))]
    [else (<= (abs (- got want)) 1e-12)]))

(define components
  '((0.4454165509112693 0.8908331018225388 0.08955713746845508)
    (-0.04005116944995218 -0.08010233889990337 0.9959816861410936)))

(define transformed
  '((2.227082754556347 -0.20025584724975892) (4.543722646581149 0.5954699916415758)
                                             (6.681248263669041 -0.6007675417492767)
                                             (8.997888155693843 0.19495829714205792)))

(test-gpu "the manual: the example's components, spectrum and transform"
  (define fitted (tsvd-fit (X) #:n-components 2))
  (check-true (close? (device-matrix->list* (tsvd-result-components fitted)) components))
  (check-true (close? (device-vector->list (tsvd-result-singular-values fitted))
                      '(12.296600453290031 0.8908520035041381)))
  (check-true (close? (device-vector->list (tsvd-result-explained-variance-ratio fitted))
                      '(0.9694773373803781 0.030522662619622817)))
  (check-true (close? (device-matrix->list* (tsvd-transform fitted (X))) transformed)))

(test-gpu "the manual: the fit's refusals, and the full round trip"
  (check-cuml-error 'logic
                    "tsvd-fit: n-components: expected at most 3, got 4"
                    (lambda () (tsvd-fit (X) #:n-components 4)))
  (check-cuml-error 'logic
                    "tsvd-fit: solver: expected full or jacobi"
                    (lambda () (tsvd-fit (X) #:solver 'arpack)))
  (define full (tsvd-fit (X) #:n-components 3))
  (check-true
   (close? (~> (tsvd-transform full (X)) (tsvd-inverse-transform full _) device-matrix->list*) rows)))

(test-gpu "the manual: tsvd-params and the estimator"
  (check-equal? (hash-ref (tsvd-params #:n-components 3) 'n-components) 3)
  (check-true (close? (device-matrix->list* (transform (fit (truncated-svd #:n-components 2) (X))
                                                       (X)))
                      transformed))
  (check-cuml-error 'logic
                    "transform: truncated-svd is not fitted"
                    (lambda () (transform (truncated-svd) (X)))))
