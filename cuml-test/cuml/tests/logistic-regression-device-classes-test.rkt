#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit logistic-regression-result-classes)
         (only-in rackunit check-equal? check-true)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix on-vector))

(define n (expt 2 18))

(define rows
  (for/list ([i (in-range n)])
    (list (exact->inexact (- (modulo (* i 7919) 1000) 500)))))

(define labels
  (for/list ([r (in-list rows)])
    (if (positive? (car r)) 3 -2)))

(define (allocated thunk)
  (collect-garbage)
  (define before (current-memory-use 'cumulative))
  (define result (thunk))
  (values result (- (current-memory-use 'cumulative) before)))

(test-gpu "a class-weight hash on device labels allocates on the host by classes, not rows"
  (define X (on-matrix rows 'float32))
  (define y (on-vector labels 'int32))
  (define (fit cw)
    (logistic-regression-fit X y #:class-weight cw #:max-iter 3 #:tol 1.0))
  (fit #f)
  (define-values (plain plain-bytes) (allocated (lambda () (fit #f))))
  (define-values (weighted weighted-bytes) (allocated (lambda () (fit (hash -2 1 3 2)))))
  (check-equal? (logistic-regression-result-classes weighted) '(-2 3))
  (check-equal? (logistic-regression-result-classes plain) '(-2 3))
  (check-true (< (- weighted-bytes plain-bytes) n)
              (format "~a bytes more than without class weights" (- weighted-bytes plain-bytes))))
