#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt" rose)
         (only-in "private/refusal-rows.rkt" check-row row))

(define refused-eta0
  '#hash((1002 . 1e-6)
         (1039 . 5e-4)
         (1113 . 2e-5)
         (1150 . 5e-4)
         (1187 . 1e-6)
         (1224 . 1e-4)
         (1335 . 1e-4)
         (1520 . 5e-5)
         (1557 . 5e-4)
         (1594 . 1e-5)
         (1631 . 2e-5)
         (1668 . 1e-4)
         (1705 . 1e-5)
         (1077 . 2e-5)
         (1114 . 5e-4)
         (1225 . 1e-4)
         (1262 . 2e-4)
         (1410 . 2e-4)
         (1447 . 1e-4)
         (1521 . 5e-4)
         (1558 . 5e-4)
         (1595 . 1e-5)
         (1632 . 5e-5)
         (1706 . 2e-5)
         (204 . 2e-4)
         (206 . 5e-4)))

(test-gpu "an l1 alpha 10 classifier fit on 2 to 10 rows of 3 columns passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))])
    (define-values (X y) (blob-rows n 3 (+ 100 n)))
    (for ([loss (in-list '(log hinge squared-loss))])
      (law-case (format "~a ~a classifier on ~a rows" dt loss n)
        (check-not-exn
         (lambda ()
           (mbsgd-classifier-fit (on-matrix X dt) y #:loss loss #:penalty 'l1 #:alpha 10)))))))

(test-gpu "an l1 alpha 10 regressor fit on 2 to 10 rows of 3 columns, 20 seeds at 2 and 3, passes at the defaults or is refused and passes at a lower eta0"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))]
         [seed (in-list (if (<= n 3)
                            (for/list ([s (in-range 20)])
                              (+ 1000 (* 37 s) n))
                            (list (+ 200 n))))])
    (define R (normal-rows n 3 seed))
    (define (fit eta0)
      (lambda ()
        (mbsgd-regressor-fit (on-matrix R dt)
                             (on-vector (reg-target R seed) dt)
                             #:penalty 'l1
                             #:alpha 10
                             #:eta0 eta0)))
    (define lower (hash-ref refused-eta0 seed #f))
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (cond
        [lower
         (check-row "mbsgd-regressor-fit" (row "eta0 0.001" (rose "0.001") (fit 1e-3)))
         (check-not-exn (fit lower))]
        [else (check-not-exn (fit 1e-3))]))))
