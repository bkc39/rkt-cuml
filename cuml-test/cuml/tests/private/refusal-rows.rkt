#lang racket/base

(require (only-in raft list*->device-matrix)
         (only-in "cuml-error.rkt" check-cuml-error))

(provide (struct-out row)
         check-row
         norm-message
         with-row)

(struct row (id message call))

(define (check-row who r [thunk (row-call r)])
  (check-cuml-error 'logic (format "~a: ~a" who (row-message r)) thunk))

(define (with-row bad #:then [then '()])
  (list*->device-matrix (append '((1 1) (2 3)) (list bad) '((4 4) (5 6)) (list '(+nan.0 0)) then)
                        #:dtype 'float32))

(define (norm-message got)
  (format "X: row 2: expected a squared norm of at most 4.253529586511731e+37, got ~a" got))
