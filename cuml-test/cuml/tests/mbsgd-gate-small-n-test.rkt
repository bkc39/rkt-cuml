#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit)
         (only-in racket/match match-define)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt" lost-at-eta0 refused-at-eta0 regressor-fit))

(define small-n-refusals
  '((204 2e-4 0.612) (206 5e-4 0.607)
                     (1002 1e-6 714)
                     (1039 5e-4 0.481)
                     (1077 2e-5 5.7)
                     (1113 2e-5 7.29)
                     (1114 5e-4 0.322)
                     (1150 5e-4 0.314)
                     (1187 1e-6 3330)
                     (1224 1e-4 2.28)
                     (1225 1e-4 1.86)
                     (1262 2e-4 1.08)
                     (1335 1e-4 3.02)
                     (1410 2e-4 0.576)
                     (1447 1e-4 1.55)
                     (1520 5e-5 4.13)
                     (1521 5e-4 0.458)
                     (1557 5e-4 0.4)
                     (1558 5e-4 0.405)
                     (1594 1e-5 20.0)
                     (1595 1e-5 15.9)
                     (1631 2e-5 7.53)
                     (1632 5e-5 3.62)
                     (1668 1e-4 1.72)
                     (1705 1e-5 9.16)
                     (1706 2e-5 4.83)))

(test-gpu "an l1 alpha 10 classifier fit on 2 to 10 rows of 3 columns passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))])
    (define-values (X y) (blob-rows n 3 (+ 100 n)))
    (for ([loss (in-list '(log hinge squared-loss))])
      (law-case (format "~a ~a classifier on ~a rows" dt loss n)
        (check-not-exn
         (lambda ()
           (mbsgd-classifier-fit (on-matrix X dt) y #:loss loss #:penalty 'l1 #:alpha 10)))))))

(define (small-n-seeds n)
  (if (<= n 3)
      (for/list ([s (in-range 20)])
        (+ 1000 (* 37 s) n))
      (list (+ 200 n))))

(define (small-n-fit dt n seed)
  (define R (normal-rows n 3 seed))
  (lambda (eta0) (regressor-fit R (reg-target R seed) dt #:penalty 'l1 #:alpha 10 #:eta0 eta0)))

(define (refusal-of seed)
  (assv seed small-n-refusals))

(define small-n-losses '((1076 5e-4 -0.027)))

(define (loss-of seed)
  (assv seed small-n-losses))

(test-gpu "an l1 alpha 10 regressor fit on 2 to 10 rows of 3 columns outside the refusal table passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))]
         [seed (in-list (small-n-seeds n))]
         #:unless (or (refusal-of seed) (loss-of seed)))
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (check-not-exn ((small-n-fit dt n seed) 1e-3)))))

(test-gpu "an l1 alpha 10 regressor fit predicting worse than the zero model is refused at the defaults"
  (for* ([dt (in-list dtypes)]
         [r (in-list small-n-losses)])
    (match-define (list seed _ _) r)
    (law-case (format "~a regressor on 2 rows, seed ~a" dt seed)
      (lost-at-eta0 (small-n-fit dt 2 seed) 1e-3))))

(test-gpu "an l1 alpha 10 regressor fit predicting worse than the zero model passes at its table eta0"
  (for* ([dt (in-list dtypes)]
         [r (in-list small-n-losses)])
    (match-define (list seed eta0 _) r)
    (law-case (format "~a regressor on 2 rows, seed ~a" dt seed)
      (check-not-exn ((small-n-fit dt 2 seed) eta0)))))

(test-gpu "an l1 alpha 10 regressor fit in the refusal table is refused at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))]
         [seed (in-list (small-n-seeds n))]
         #:when (refusal-of seed))
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (refused-at-eta0 (small-n-fit dt n seed) 1e-3))))

(test-gpu "an l1 alpha 10 regressor fit in the refusal table passes at its table eta0"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))]
         [seed (in-list (small-n-seeds n))]
         #:when (refusal-of seed))
    (match-define (list _ eta0 _) (refusal-of seed))
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (check-not-exn ((small-n-fit dt n seed) eta0)))))
