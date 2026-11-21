#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit logistic-regression-result-classes)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" on-matrix))

(define n (expt 2 18))

(define column
  (for/list ([i (in-range n)])
    (exact->inexact (- (modulo (* i 7919) 1000) 500))))

(define labels
  (for/list ([v (in-list column)])
    (if (positive? v) 3 -2)))

(define (call-with-allocation-count thunk)
  (collect-garbage)
  (define before (current-memory-use 'cumulative))
  (define result (thunk))
  (values result (- (current-memory-use 'cumulative) before)))

(test-gpu "device labels: class weights sized by classes, not rows"
  (define X (on-matrix (map list column) 'float32))
  (define y (on-vector labels 'int32))
  (define (fit cw)
    (logistic-regression-fit X y #:class-weight cw #:max-iter 3 #:tol 1.0))
  (define warmed (fit #f))
  (define-values (plain plain-bytes) (call-with-allocation-count (lambda () (fit #f))))
  (define-values (weighted weighted-bytes)
    (call-with-allocation-count (lambda () (fit (hash -2 1 3 2)))))
  (test-case "the warm-up fit reads the classes"
    (check-equal? (logistic-regression-result-classes warmed) '(-2 3)))
  (test-case "the weighted fit reads the classes from the shim"
    (check-equal? (logistic-regression-result-classes weighted) '(-2 3)))
  (test-case "the plain fit reads the same classes"
    (check-equal? (logistic-regression-result-classes plain) '(-2 3)))
  (test-case "the hash adds fewer host bytes than y has rows"
    (check-true (< (- weighted-bytes plain-bytes) n)
                (format "~a bytes more than without class weights" (- weighted-bytes plain-bytes)))))
