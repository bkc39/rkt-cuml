#lang racket/base

(require (only-in racket/list filter-not partition)
         (only-in racket/match match-define match-lambda)
         (only-in rackunit check-= check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt"
                  at-eta0
                  classifier-fit
                  loss-r2
                  lost-at-eta0
                  refused-at-eta0))

(define small-n-refusals
  '((204 2e-4) (206 5e-4)
               (1002 1e-6)
               (1039 5e-4)
               (1077 2e-5)
               (1113 2e-5)
               (1114 5e-4)
               (1150 5e-4)
               (1187 1e-6)
               (1224 1e-4)
               (1225 1e-4)
               (1262 2e-4)
               (1335 1e-4)
               (1410 2e-4)
               (1447 1e-4)
               (1520 5e-5)
               (1521 5e-4)
               (1557 5e-4)
               (1558 5e-4)
               (1594 1e-5)
               (1595 1e-5)
               (1631 2e-5)
               (1632 5e-5)
               (1668 1e-4)
               (1705 1e-5)
               (1706 2e-5)))

(test-gpu "an l1 alpha 10 classifier fit on 2 to 10 rows of 3 columns passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [n (in-list '(2 3 4 6 10))])
    (define-values (X y) (blob-rows n 3 (+ 100 n)))
    (for ([loss (in-list '(log hinge squared-loss))])
      (law-case (format "~a ~a classifier on ~a rows" dt loss n)
        (check-not-exn (classifier-fit X y dt #:loss loss #:penalty 'l1 #:alpha 10))))))

(define (small-n-seeds n)
  (if (<= n 3)
      (for/list ([s (in-range 20)])
        (+ 1000 (* 37 s) n))
      (list (+ 200 n))))

(define (small-n-fit dt n seed)
  (define R (normal-rows n 3 seed))
  (at-eta0 R (reg-target R seed) dt #:penalty 'l1 #:alpha 10))

(define small-n-losses '((1076 5e-4 -0.027)))

(define small-n-trials
  (for*/list ([n (in-list '(2 3 4 6 10))]
              [seed (in-list (small-n-seeds n))])
    (list n seed)))

(define-values (refused-trials unrefused-trials)
  (partition (match-lambda
               [(list _ seed) (assv seed small-n-refusals)])
             small-n-trials))

(define passing-trials
  (filter-not (match-lambda
                [(list _ seed) (assv seed small-n-losses)])
              unrefused-trials))

(test-gpu "an l1 alpha 10 regressor fit on 2 to 10 rows of 3 columns outside the refusal table passes at the defaults"
  (for* ([dt (in-list dtypes)]
         [t (in-list passing-trials)])
    (match-define (list n seed) t)
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (check-not-exn ((small-n-fit dt n seed) 1e-3)))))

(test-gpu "an l1 alpha 10 regressor fit predicting worse than the zero model is refused at the defaults"
  (for* ([dt (in-list dtypes)]
         [r (in-list small-n-losses)])
    (match-define (list seed _ r2) r)
    (law-case (format "~a regressor on 2 rows, seed ~a" dt seed)
      (lost-at-eta0 (small-n-fit dt 2 seed) 1e-3)
      (check-= (loss-r2 ((small-n-fit dt 2 seed) 1e-3)) r2 0.01))))

(test-gpu "an l1 alpha 10 regressor fit predicting worse than the zero model passes at its table eta0"
  (for* ([dt (in-list dtypes)]
         [r (in-list small-n-losses)])
    (match-define (list seed eta0 _) r)
    (law-case (format "~a regressor on 2 rows, seed ~a" dt seed)
      (check-not-exn ((small-n-fit dt 2 seed) eta0)))))

(test-gpu "an l1 alpha 10 regressor fit in the refusal table is refused at the defaults"
  (for* ([dt (in-list dtypes)]
         [t (in-list refused-trials)])
    (match-define (list n seed) t)
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (refused-at-eta0 (small-n-fit dt n seed) 1e-3))))

(test-gpu "an l1 alpha 10 regressor fit in the refusal table passes at its table eta0"
  (for* ([dt (in-list dtypes)]
         [t (in-list refused-trials)])
    (match-define (list n seed) t)
    (match-define (list _ eta0) (assv seed small-n-refusals))
    (law-case (format "~a regressor on ~a rows, seed ~a" dt n seed)
      (check-not-exn ((small-n-fit dt n seed) eta0)))))
