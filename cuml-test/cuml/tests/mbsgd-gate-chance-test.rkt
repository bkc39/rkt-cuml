#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit)
         (only-in racket/match match-define)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/mbsgd-data.rkt" blob-rows)
         (only-in "private/mbsgd-gate-data.rkt" blobs labels rose)
         (only-in "private/refusal-rows.rkt" check-row row))

(define-values (overlap-512 overlap-512-labels) (blob-rows 512 6 45 #:sep 0.3))

(define chance-fits
  `(("hinge l2 alpha 0.1 on overlapping blobs" ,overlap-512 ,overlap-512-labels hinge l2 0.01)
    ("log l1 alpha 0.1 on blobs" ,blobs ,labels log l1 0.1)
    ("hinge l1 alpha 0.1 on blobs" ,blobs ,labels hinge l1 0.05)))

(test-gpu "a one-epoch fit worse than the zero model is refused; eta0/10 passes"
  (for* ([dt (in-list dtypes)]
         [c (in-list chance-fits)])
    (match-define (list name X y loss penalty eta0) c)
    (define (go step)
      (lambda ()
        (mbsgd-classifier-fit (on-matrix X dt)
                              y
                              #:loss loss
                              #:penalty penalty
                              #:alpha 0.1
                              #:eta0 step
                              #:batch-size 1
                              #:epochs 1)))
    (law-case (format "~a ~a" dt name)
      (check-row "mbsgd-classifier-fit" (row name (rose (number->string eta0)) (go eta0)))
      (check-not-exn (go (/ eta0 10))))))

(test-gpu "an l1 rise above five times the zero model is refused however large eta0 makes the allowance"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a log l1 alpha 10 on blobs, eta0 0.3" dt)
      (check-row "mbsgd-classifier-fit"
                 (row "log l1 alpha 10 on blobs, eta0 0.3"
                      (rose "0.3")
                      (lambda ()
                        (mbsgd-classifier-fit (on-matrix blobs dt)
                                              labels
                                              #:loss 'log
                                              #:penalty 'l1
                                              #:alpha 10
                                              #:eta0 0.3
                                              #:learning-rate 'constant
                                              #:batch-size 1
                                              #:epochs 1)))))))
