#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in rackunit check-equal?)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt"
                  learning-rates
                  normal-rows
                  penalties
                  reg-parts
                  reg-target
                  tiled-rows))

(define rows (normal-rows 300 5 21))
(define targets (reg-target rows 21))

(define (twice fit)
  (check-equal? (fit) (fit)))

(test-gpu "a fit repeats bit for bit on the twin grid"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))]
         [lr (in-list learning-rates)]
         [penalty (in-list penalties)]
         [seed (in-list '(#f 7))])
    (law-case (format "~a fit-intercept ~a ~a ~a random-state ~a" dt intercept? lr penalty seed)
      (define X (on-matrix rows dt))
      (define y (on-vector targets dt))
      (twice (lambda ()
               (reg-parts (mbsgd-regressor-fit X
                                               y
                                               #:fit-intercept intercept?
                                               #:learning-rate lr
                                               #:penalty penalty
                                               #:batch-size 16
                                               #:eta0 0.01
                                               #:epochs 50
                                               #:random-state seed)))))))

(test-gpu "a large fit repeats bit for bit"
  (for* ([shape (in-list '((200000 20 256) (1000000 20 4096)))]
         [dt (in-list dtypes)])
    (define-values (n d batch) (apply values shape))
    (define X-rows (tiled-rows n d 61))
    (define X (on-matrix X-rows dt))
    (define y (on-vector (reg-target X-rows 61) dt))
    (for ([seed (in-list '(#f 7))])
      (law-case (format "~ax~a batch ~a ~a random-state ~a" n d batch dt seed)
        (twice (lambda ()
                 (reg-parts (mbsgd-regressor-fit X y #:batch-size batch #:epochs 5 #:random-state seed))))))))
