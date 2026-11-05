#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  algorithms
                  fit-of
                  insert-column
                  kept-rows
                  lcase-X
                  lcase-w
                  lcase-y
                  normal-case
                  remove-at
                  zeroed)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define wide (normal-case 5003 41 3 #:weighted? #t))
(define gone '(0 7 2047 2048 2049 4095 5002))

(test-gpu "past one row chunk and one column tile, zero weights delete rows and constants drop out"
  (for* ([setting (in-list '((float64 #t) (float32 #f)))]
         #:do [(match-define (list dt fi) setting)
               (define X (kept-rows (lcase-X wide) gone))
               (define y (kept-rows (lcase-y wide) gone))
               (define w (kept-rows (lcase-w wide) gone))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a ~a" dt fi a)
      (define (fit X y w)
        (fit-of X y w dt #:algorithm a #:fit-intercept fi))
      (define deleted (fit X y w))
      (check-true (bits-equal? (fit (lcase-X wide) (lcase-y wide) (zeroed (lcase-w wide) gone))
                               deleted)
                  "zero weights at the chunk edges are deleted rows")
      (for ([at (in-list '(0 20 41))])
        (match-define (list coef b) (fit (insert-column X at (lambda (_) (if fi 7 0))) y w))
        (check-equal? (list-ref coef at) 0.0)
        (check-true (bits-equal? (list (remove-at coef at) b) deleted)
                    (format "a constant column at ~a leaves the rest" at))))))
