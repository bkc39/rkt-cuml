#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target))

(test-gpu "an l1 alpha 10 fit on 2 to 10 rows of 3 columns, 20 seeds at 2 and 3, passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))])
    (define-values (X y) (blob-rows n 3 (+ 100 n)))
    (for ([loss (in-list '(log hinge squared-loss))])
      (law-case (format "~a ~a classifier on ~a rows" dt loss n)
        (check-not-exn
         (lambda () (mbsgd-classifier-fit (on-matrix X dt) y #:loss loss #:penalty 'l1 #:alpha 10)))))
    (define seeds
      (if (<= n 3)
          (for/list ([s (in-range 20)])
            (+ 1000 (* 37 s) n))
          (list (+ 200 n))))
    (for ([seed (in-list seeds)])
      (define R (normal-rows n 3 seed))
      (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
        (check-not-exn (lambda ()
                         (mbsgd-regressor-fit (on-matrix R dt)
                                              (on-vector (reg-target R seed) dt)
                                              #:penalty 'l1
                                              #:alpha 10)))))))
