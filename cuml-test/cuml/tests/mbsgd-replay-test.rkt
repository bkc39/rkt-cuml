#lang racket/base

(require (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-not-equal? check-true)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" apply-keywords)
         (only-in "private/linear-data.rkt" by-dtype dtypes law-case)
         (only-in "private/mbsgd-data.rkt"
                  blob-rows
                  coef-gap
                  learning-rates
                  normal-rows
                  penalties
                  reg-target)
         (only-in "private/mbsgd-twin-data.rkt" clf-fit reg-fit)
         (only-in "private/sgd-replay.rkt" replay sgd-replay))

(define rows (normal-rows 96 3 9))
(define targets (reg-target rows 9 #:noise 0.3))
(define-values (blobs labels) (blob-rows 64 3 1))

(define stop-margin 1e-6)

(define (replayed settings X y)
  (apply-keywords sgd-replay (hash-remove settings '#:shuffle) (list X y)))

(define (replay-gap ours coef intercept)
  (match-define (list c b) ours)
  (max (coef-gap c coef) (/ (abs (- b intercept)) (max (abs intercept) 1.0))))

(define (check-replay dt ours r)
  (match-define (struct* replay ([coef coef] [intercept intercept] [margin margin])) r)
  (check-true (> margin stop-margin) (format "stop-test margin ~a" margin))
  (define gap (replay-gap ours coef intercept))
  (check-true (<= gap (by-dtype dt 5e-6 1e-13)) (format "gap ~a" gap)))

(define (h1-settings lr penalty size tol)
  (hash '#:penalty penalty
        '#:alpha 0.01
        '#:learning-rate lr
        '#:eta0 0.05
        '#:batch-size size
        '#:tol tol
        '#:epochs 200
        '#:shuffle #f))

(test-gpu "the device schedule is the host replay of sec. 2.2-2.3 in float64"
  (for* ([lr (in-list learning-rates)]
         [penalty (in-list penalties)]
         [size (in-list '(96 128 20))]
         [tol (in-list '(0.0 1e-3))])
    (define s (h1-settings lr penalty size tol))
    (define r (replayed s rows targets))
    (for ([dt (in-list dtypes)])
      (law-case (format "~a ~a ~a bs ~a tol ~a" dt lr penalty size tol)
        (check-replay dt (reg-fit dt rows targets s) r)))))

(test-gpu "log and -1/+1 hinge without an intercept follow the host replay"
  (for ([loss (in-list '(log hinge))])
    (define s
      (hash '#:alpha 0.01
            '#:fit-intercept #f
            '#:eta0 0.1
            '#:batch-size 16
            '#:epochs 100
            '#:tol 0.0
            '#:shuffle #f))
    (define target
      (if (eq? loss 'hinge)
          (map (lambda (c) (- (* 2 c) 1)) labels)
          labels))
    (define r (replayed (hash-set s '#:loss loss) blobs target))
    (for ([dt (in-list dtypes)])
      (law-case (format "~a ~a" dt loss)
        (check-replay dt (clf-fit dt blobs labels loss s) r)))))

(test-gpu "the fit stops at the epoch the replay stops at"
  (for* ([lr (in-list learning-rates)]
         [size (in-list '(96 20))])
    (define s (h1-settings lr 'l2 size 1e-3))
    (match-define (struct* replay ([epochs m] [margin margin])) (replayed s rows targets))
    (for ([dt (in-list dtypes)])
      (law-case (format "~a ~a bs ~a stops after ~a epochs" dt lr size m)
        (check-true (> margin stop-margin) (format "stop-test margin ~a" margin))
        (check-true (< m 200) "the replay stops early")
        (define (at epochs)
          (reg-fit dt rows targets (hash-set s '#:epochs epochs)))
        (check-equal? (at m) (at (+ m 50)))
        (check-not-equal? (at (sub1 m)) (at m))))))
