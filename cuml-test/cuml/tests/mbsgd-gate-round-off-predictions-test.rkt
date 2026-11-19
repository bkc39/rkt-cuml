#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-result-coef mbsgd-regressor-result-intercept)
         (only-in racket/flonum flsingle)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt"
                  constant-rows
                  gate-floor-share
                  gate-round-off-factor
                  gate-round-off-start
                  off-predictions-at-eta0
                  regressor-fit
                  round-off-prediction-share))

(define (mean xs)
  (/ (apply + xs) (length xs)))

(define half (normal-rows 150 4 3))
(define zero-mean-rows (append half (map (lambda (r) (map - r)) half)))

(define standardised-rows
  (let* ([columns (apply map list constant-rows)]
         [means (map mean columns)]
         [sds (for/list ([c (in-list columns)]
                         [m (in-list means)])
                (sqrt (mean (map (lambda (v) (sqr (- v m))) c))))])
    (for/list ([r (in-list constant-rows)])
      (map (lambda (v m s) (/ (- v m) s)) r means sds))))

(define off-the-constant
  '((zero-mean l1 10.0) (zero-mean l1 100.0)
                        (zero-mean elasticnet 1.0)
                        (standardised l1 1.0)
                        (standardised l1 10.0)
                        (standardised l1 100.0)))

(test-gpu "a constant target's fit whose predictions are off by more than a share of it is refused"
  (for* ([dt (in-list dtypes)]
         [r (in-list off-the-constant)])
    (match-define (list xs penalty alpha) r)
    (law-case (format "~a ~a y = 1.7 ~a alpha ~a" dt xs penalty alpha)
      (off-predictions-at-eta0
       (lambda (eta0)
         (regressor-fit (if (eq? xs 'zero-mean) zero-mean-rows standardised-rows)
                        (make-list 300 1.7)
                        dt
                        #:penalty penalty
                        #:alpha alpha
                        #:eta0 eta0))
       0.1))))

(test-gpu "a small constant's fit at the defaults is refused by its predictions"
  (law-case "float64 standardised y = 1.1e-4 l1"
    (off-predictions-at-eta0
     (lambda (eta0)
       (regressor-fit standardised-rows (make-list 300 1.1e-4) 'float64 #:penalty 'l1 #:eta0 eta0))
     0.001)))

(define (mean-square xs)
  (mean (map sqr xs)))

(define (residuals fit X y)
  (define w (device-vector->list (mbsgd-regressor-result-coef fit)))
  (define b (mbsgd-regressor-result-intercept fit))
  (for/list ([r (in-list X)]
             [t (in-list y)])
    (- t b (apply + (map * r w)))))

(define (dtype-epsilon dt)
  (if (eq? dt 'float32)
      (expt 2.0 -23)
      (expt 2.0 -52)))

(define (near-constant-target dt rows noise offset s)
  (for/list ([r (in-list rows)]
             [e (in-list noise)])
    (define v (+ offset (* s (car r)) (* 0.2 s (car e))))
    (if (eq? dt 'float32)
        (flsingle v)
        v)))

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
           (* gate-round-off-factor (sqr (dtype-epsilon dt)) (mean-square y))))
      (define loss
        (mean-square (residuals ((regressor-fit X y dt #:penalty 'l1 #:alpha alpha #:eta0 0.2)) X y)))
      (check < (- 1 (/ loss start)) -1)
      (check <= start (* (cdr (assq dt gate-round-off-start)) round-off-floor))
      (check <= (/ (sqrt loss) (abs centre)) round-off-prediction-share))))
