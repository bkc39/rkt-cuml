#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit)
         (only-in cuml/linear-model/private/labels current-label-classes-job)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-matrix device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix on-vector))

(define rows '((0.0) (1.0) (2.0) (3.0) (4.0) (5.0) (6.0) (7.0)))
(define labels '(0 0 1 0 1 1 0 1))

(define fitted-rows `((#f "accepted" 0) (balanced "accepted" 0) (,(hash 0 1.0 1 2.0) "accepted" 1)))

(define (outcome-and-jobs thunk)
  (define jobs 0)
  (define job (current-label-classes-job))
  (define outcome
    (parameterize ([current-label-classes-job (lambda args
                                                (set! jobs (add1 jobs))
                                                (apply job args))])
      (with-handlers ([exn:fail? exn-message])
        (thunk)
        "accepted")))
  (list outcome jobs))

(test-gpu "device labels: a label-classes job only for a class-weight hash"
  (define X (on-matrix rows 'float64))
  (define y (on-vector labels 'int32))
  (for ([row (in-list fitted-rows)])
    (match-define (list cw outcome jobs) row)
    (test-case (format "class-weight ~a" cw)
      (check-equal? (outcome-and-jobs (lambda () (logistic-regression-fit X y #:class-weight cw)))
                    (list outcome jobs)))))

(define boundary-rows
  `((0 () ,(hash 0 1.0) "logistic-regression-fit: X: expected at least 2 rows, got 0" 0)
    (1 (5)
       ,(hash 0 1.0)
       "logistic-regression-fit: class-weight: classes (5) are not in class-weight"
       1)
    (2 (0 1) ,(hash 0 1.0 1 5.0) "accepted" 1)))

(test-gpu "device labels: a class-weight hash on 0, 1 and 2 rows"
  (for ([row (in-list boundary-rows)])
    (match-define (list n ys cw outcome jobs) row)
    (test-case (format "~a rows" n)
      (define X
        (if (zero? n)
            (device-matrix 0 1 #:dtype 'float32)
            (on-matrix (for/list ([i (in-range n)])
                         (list (exact->inexact i)))
                       'float32)))
      (define y
        (if (zero? n)
            (device-vector 0 #:dtype 'int32)
            (on-vector ys 'int32)))
      (check-equal? (outcome-and-jobs (lambda () (logistic-regression-fit X y #:class-weight cw)))
                    (list outcome jobs)))))
