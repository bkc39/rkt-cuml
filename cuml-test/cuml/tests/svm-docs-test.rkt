#lang racket/base

(require (only-in cuml fit score svc)
         (only-in cuml/svm
                  svc-decision-function
                  svc-fit
                  svc-predict
                  svc-result-n-support
                  svc-result-support)
         (only-in rackunit check-equal?)
         (only-in raft device-vector->list list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define (X)
  (list*->device-matrix '((0.0 0.0) (0.0 1.0) (1.0 0.0) (3.0 3.0) (3.0 4.0) (4.0 3.0))
                        #:dtype 'float64))

(define y '(a a a b b b))

(test-gpu "the manual: svc-fit's support, decisions and labels"
  (define r (svc-fit (X) y #:kernel 'linear))
  (check-equal? (svc-result-n-support r) '(2 1))
  (check-equal? (device-vector->list (svc-result-support r)) '(1 2 3))
  (check-equal? (device-vector->list (svc-decision-function r (X))) '(-1.4000236012070328 -1.0000590030175827 -0.9999704984912083 1.0000295015087912 1.3999940996982414 1.4000826042246157))
  (check-equal? (svc-predict r (X)) y))

(test-gpu "the manual: svc-fit's refusals"
  (check-cuml-error 'logic
                    "svc-fit: y: expected two classes, got one ('a)"
                    (lambda () (svc-fit (X) '(a a a a a a))))
  (check-cuml-error 'logic
                    "svc-fit: gamma: expected 'scale, 'auto or a finite real at least 0, got -1.0"
                    (lambda () (svc-fit (X) y #:gamma -1.0))))

(test-gpu "the manual: the estimator scores its training rows"
  (check-equal? (score (fit (svc #:kernel 'linear) (X) y) (X) y) 1.0))
