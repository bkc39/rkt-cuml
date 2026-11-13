#lang racket/base

(require (only-in cuml fit logistic-regression logistic-regression-model predict score)
         (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-predict
                  logistic-regression-result-classes
                  logistic-regression-result-coef)
         (only-in racket/list count)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" blob-case lr-case-X lr-case-y))

(define binary (blob-case "binary" 80 3 2 11))
(define three (blob-case "three classes" 90 3 3 12 #:sep 3.0))
(define X (on-matrix (lr-case-X binary) 'float64))
(define exact-y (lr-case-y binary))

(define (mixed ys)
  (for/list ([v (in-list ys)]
             [i (in-naturals)])
    (if (even? i)
        v
        (exact->inexact v))))

(test-gpu "host real labels score and predict the same given exact or as flonums"
  (define e (fit (logistic-regression) X exact-y))
  (check-equal? (score e X (map exact->inexact exact-y)) (score e X exact-y))
  (check-equal? (logistic-regression-result-classes (logistic-regression-model e)) '(0 1)))

(test-gpu "a y mixing 0 and 0.0 has two flonum classes and scores as the exact y"
  (define e (fit (logistic-regression) X (mixed exact-y)))
  (check-equal? (logistic-regression-result-classes (logistic-regression-model e)) '(0.0 1.0))
  (check-equal? (score e X (mixed exact-y)) (score (fit (logistic-regression) X exact-y) X exact-y))
  (check-equal? (predict e X) (map exact->inexact (predict (fit (logistic-regression) X exact-y) X))))

(test-gpu "exact labels 7, 2^53 and 2^53+1 are three classes, each predicted"
  (define big (expt 2 53))
  (define labels (list 7 big (add1 big)))
  (define y (map (lambda (c) (list-ref labels c)) (lr-case-y three)))
  (define r (logistic-regression-fit (on-matrix (lr-case-X three) 'float64) y))
  (check-equal? (logistic-regression-result-classes r) labels)
  (check-equal? (length (device-matrix->list* (logistic-regression-result-coef r))) 3)
  (define predicted (logistic-regression-predict r (on-matrix (lr-case-X three) 'float64)))
  (for ([c (in-list labels)])
    (test-case (format "~a" c)
      (check-true (positive? (count (lambda (v) (equal? v c)) predicted))))))
