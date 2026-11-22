#lang racket/base

(require (only-in cuml svc svc-model svc?)
         (only-in cuml/estimator
                  decision-function
                  estimator-params
                  estimator-set-params
                  fit
                  fitted?
                  predict
                  score)
         (only-in cuml/svm svc-decision-function svc-fit svc-predict)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs fit-parts))

(test-case "parameters round-trip and print"
  (define e (svc #:C 2.5 #:kernel 'poly #:degree 2 #:gamma 'auto))
  (check-true (svc? e))
  (check-false (fitted? e))
  (check-equal? (hash-ref (estimator-params e) 'C) 2.5)
  (check-equal? (hash-ref (estimator-params e) 'kernel) 'poly)
  (check-equal? (hash-ref (estimator-params e) 'gamma) 'auto)
  (check-equal? (estimator-params (estimator-set-params e (estimator-params e))) (estimator-params e))
  (check-equal? (hash-ref (estimator-params (estimator-set-params e (hash 'tol 1e-4))) 'tol) 1e-4)
  (check-equal? (format "~v" (svc #:C 2.5)) "#<svc #:C 2.5>"))

(define-values (rows labels) (dyadic-blobs 120 3 6))

(test-gpu "fit, predict, decision and score equal the procedures"
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define X (on-matrix rows dt))
      (define w
        (on-vector (for/list ([i (in-range 120)])
                     (if (odd? i) 2.0 1.0))
                   dt))
      (define e (fit (svc #:kernel 'rbf #:C 2.0) X labels #:sample-weight w))
      (define r (svc-fit X labels #:kernel 'rbf #:C 2.0 #:sample-weight w))
      (check-true (fitted? e))
      (check-equal? (fit-parts (svc-model e)) (fit-parts r))
      (check-equal? (predict e X) (svc-predict r X))
      (check-equal? (device-vector->list (decision-function e X))
                    (device-vector->list (svc-decision-function r X)))
      (check-equal? (score e X labels)
                    (exact->inexact (/ (for/sum ([p (in-list (svc-predict r X))] [l (in-list labels)])
                                                (if (equal? p l) 1 0))
                                       120)))
      (check-equal?
       (fit-parts (svc-model (fit (svc) X labels #:sample-weight (on-vector (make-list 120 1.0) dt))))
       (fit-parts (svc-model (fit (svc) X labels)))))))
