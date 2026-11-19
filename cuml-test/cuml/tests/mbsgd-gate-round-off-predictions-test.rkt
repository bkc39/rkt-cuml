#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-result-coef mbsgd-regressor-result-intercept)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case rounded unit-roundoff)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt"
                  at-eta0
                  constant-rows
                  gate-floor-share
                  gate-round-off-factor
                  gate-round-off-start
                  off-predictions-at-eta0
                  regressor-fit
                  round-off-prediction-share
                  zero-mean-rows)
         (only-in "private/metrics-data.rkt" mean))

(define standardised-rows
  (let* ([columns (apply map list constant-rows)]
         [means (map mean columns)]
         [sds (for/list ([c (in-list columns)]
                         [m (in-list means)])
                (sqrt (mean (map (lambda (v) (sqr (- v m))) c))))])
    (for/list ([r (in-list constant-rows)])
      (map (lambda (v m s) (/ (- v m) s)) r means sds))))

(define off-the-constant
  `((zero-mean ,zero-mean-rows l1 10.0) (zero-mean ,zero-mean-rows l1 100.0)
                                        (zero-mean ,zero-mean-rows elasticnet 1.0)
                                        (standardised ,standardised-rows l1 1.0)
                                        (standardised ,standardised-rows l1 10.0)
                                        (standardised ,standardised-rows l1 100.0)))

(test-gpu "a constant target's fit whose predictions are off by more than a share of it is refused"
  (for* ([dt (in-list dtypes)]
         [r (in-list off-the-constant)])
    (match-define (list name X penalty alpha) r)
    (law-case (format "~a ~a y = 1.7 ~a alpha ~a" dt name penalty alpha)
      (off-predictions-at-eta0 (at-eta0 X (make-list 300 1.7) dt #:penalty penalty #:alpha alpha)
                               0.1))))

(test-gpu "a small constant's fit at the defaults is refused by its predictions"
  (law-case "float64 standardised y = 1.1e-4 l1"
    (off-predictions-at-eta0 (at-eta0 standardised-rows (make-list 300 1.1e-4) 'float64 #:penalty 'l1)
                             0.001)))

(define (mean-square xs)
  (mean (map sqr xs)))

(define (residuals fit X y)
  (define w (device-vector->list (mbsgd-regressor-result-coef fit)))
  (define b (mbsgd-regressor-result-intercept fit))
  (for/list ([r (in-list X)]
             [t (in-list y)])
    (- t b (apply + (map * r w)))))

(define (near-constant-target dt rows noise offset s)
  (for/list ([r (in-list rows)]
             [e (in-list noise)])
    (rounded dt (+ offset (* s (car r)) (* 0.2 s (car e))))))

(define within-ulps
  '((float32 200 3 9 1e6 0.36 1.44)
    (float64 50 53 59 1e10 2.6626230610492523e-5 1.065049224419701e-4)))

(test-gpu "a fit at round-off within a share of the constant passes however low its R²"
  (for ([r (in-list within-ulps)])
    (match-define (list dt n x-seed e-seed offset s alpha) r)
    (law-case (format "~a n ~a y = ~a + ~a·x₁ + noise" dt n offset s)
      (define X (normal-rows n 4 x-seed))
      (define y (near-constant-target dt X (normal-rows n 1 e-seed) offset s))
      (define centre (mean y))
      (define start (mean-square (map (lambda (t) (- t centre)) y)))
      (define round-off-floor
        (+ (* gate-floor-share start)
           (* gate-round-off-factor (sqr (* 2 (unit-roundoff dt))) (mean-square y))))
      (define loss
        (mean-square (residuals ((regressor-fit X y dt #:penalty 'l1 #:alpha alpha #:eta0 0.2)) X y)))
      (check < (- 1 (/ loss start)) -1)
      (check <= start (* (cdr (assq dt gate-round-off-start)) round-off-floor))
      (check <= (/ (sqrt loss) (abs centre)) round-off-prediction-share))))
