#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in racket/match match-define)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/logistic-data.rkt" lr-case-X)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows)
         (only-in "private/real-logistic-data.rkt" labelled labelled-set))

(define sparse-rows (normal-rows 500 5 3))
(define sparse-targets
  (for/list ([r (in-list sparse-rows)]
             [e (in-list (normal-rows 500 1 4))])
    (+ 1.0 (* 0.3 (car r)) (car e))))
(define-values (overlap overlap-labels) (blob-rows 300 4 5 #:sep 0.3))

(test-gpu "an l1 or elastic-net fit whose optimum is zero passes at the default eta0"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l1 elasticnet))])
    (for ([alpha (in-list '(1.0 5.0))])
      (law-case (format "~a ~a regressor alpha ~a" dt penalty alpha)
        (check-not-exn (lambda ()
                         (mbsgd-regressor-fit (on-matrix sparse-rows dt)
                                              (on-vector sparse-targets dt)
                                              #:penalty penalty
                                              #:alpha alpha
                                              #:l1-ratio 0.9)))))
    (for ([loss (in-list '(hinge log))])
      (law-case (format "~a ~a ~a classifier" dt penalty loss)
        (check-not-exn (lambda ()
                         (mbsgd-classifier-fit (on-matrix overlap dt)
                                               overlap-labels
                                               #:loss loss
                                               #:penalty penalty
                                               #:alpha 2.0
                                               #:l1-ratio 0.9)))))))

(test-gpu "standardised breast cancer, 30 columns, passes an l1 alpha 5 or 10 fit at the defaults"
  (match-define (labelled bc bc-labels _)
    (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t))
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(log hinge))]
         [alpha (in-list '(5 10))])
    (law-case (format "~a ~a alpha ~a" dt loss alpha)
      (check-not-exn (lambda ()
                       (mbsgd-classifier-fit (on-matrix (lr-case-X bc) dt)
                                             bc-labels
                                             #:loss loss
                                             #:penalty 'l1
                                             #:alpha alpha))))))
