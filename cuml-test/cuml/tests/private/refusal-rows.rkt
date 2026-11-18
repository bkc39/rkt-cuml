#lang racket/base

(require (only-in raft device-matrix device-vector list*->device-matrix)
         (only-in "cuml-error.rkt" check-cuml-error prefixed))

(provide (struct-out row)
         bad-X-rows
         check-row
         float-most
         grid
         index-message
         norm-message
         other
         with-bad-row
         with-row)

(struct row (id message call))

(define (check-row name r [thunk (row-call r)])
  (define message (row-message r))
  (check-cuml-error 'logic
                    (if (regexp? message)
                        (prefixed (format "~a: " name) message)
                        (format "~a: ~a" name message))
                    thunk))

(define (with-row bad #:then [then '()])
  (list*->device-matrix (append '((1 1) (2 3)) (list bad) '((4 4) (5 6)) (list '(+nan.0 0)) then)
                        #:dtype 'float32))

(define (norm-message got)
  (format "X: row 2: expected a squared norm of at most 4.253529586511731e+37, got ~a" got))

(define (grid dt [n 10])
  (list*->device-matrix (for/list ([i (in-range n)])
                          (list (add1 i) (* 2 i)))
                        #:dtype dt))

(define (other dt)
  (if (eq? dt 'float32) 'float64 'float32))

(define (with-bad-row bad)
  (with-row bad #:then '((7 7) (8 9) (9 1) (3 3))))

(define (float-most name got)
  (format "~a: expected at most 3.4028234663852886e+38, got ~a" name got))

(define (bad-X-rows)
  (list
   (row "f1" "X: expected float32 or float64, got int32" (device-matrix 4 3 #:dtype 'int32))
   (row "f2" "X: expected rank 2, got rank 1" (device-vector 4 #:dtype 'float32))
   (row "f3"
        "X: expected row-major, got col-major 4x3"
        (device-matrix 4 3 #:dtype 'float32 #:layout 'col-major))
   (row "f1 before f2" "X: expected float32 or float64, got int32" (device-vector 4 #:dtype 'int32))))

(define (index-message name entry got)
  (format "~a: entry ~a: expected an index in [0, 10), got ~a" name entry got))
