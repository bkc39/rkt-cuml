#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-not-equal?)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" scaled-rows)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" blob-rows dtype-penalties normal-rows reg-target with-ones)
         (only-in "private/mbsgd-twin-data.rkt" clf-fit overridden reg-fit))

(define rows (normal-rows 300 5 21))
(define targets (reg-target rows 21))
(define-values (blobs labels) (blob-rows 300 4 22))
(define-values (overlapping overlapping-labels) (blob-rows 300 4 23 #:sep 0.5))

(define law-settings (hash '#:batch-size 16 '#:eta0 0.01 '#:epochs 50))

(define (reg dt X y [more (hash)] #:random-state [seed #f])
  (reg-fit dt X y (overridden law-settings more) #:random-state seed))

(test-gpu "without shuffle the random state changes nothing"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (define fits
        (for/list ([seed (in-list (list #f 0 7 4294967295))])
          (reg dt rows targets (hash '#:shuffle #f) #:random-state seed)))
      (match-define (cons first others) fits)
      (for ([f (in-list others)])
        (check-equal? f first)))))

(define (scale-alpha penalty k)
  (case penalty
    [(l2) (expt 4 k)]
    [(l1) (expt 2 k)]
    [else 1]))

(test-gpu "X times 2^k with alpha and eta0 rescaled gives coef times 2^-k and the same intercept"
  (for* ([dt (in-list dtypes)]
         [lr (in-list '(constant invscaling))]
         [penalty (in-list dtype-penalties)]
         [intercept? (in-list '(#t #f))])
    (define (at k)
      (reg dt
           (scaled-rows rows (expt 2 k))
           targets
           (hash '#:penalty penalty
                 '#:alpha (* 0.01 (scale-alpha penalty k))
                 '#:eta0 (* 0.01 (expt 4 (- k)))
                 '#:learning-rate lr
                 '#:fit-intercept intercept?)))
    (match-define (list coef intercept) (at 0))
    (for ([k (in-list '(-20 -3 5 30))])
      (law-case (format "~a ~a ~a fit-intercept ~a, k ~a" dt lr penalty intercept? k)
        (check-equal? (at k) (list (map (lambda (c) (* c (expt 2 (- k)))) coef) intercept)))))
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(log hinge))])
    (define (at k)
      (clf-fit dt
               (scaled-rows blobs (expt 2 k))
               labels
               loss
               (overridden law-settings
                           (hash '#:fit-intercept #f
                                 '#:alpha (* 0.01 (expt 4 k))
                                 '#:eta0 (* 0.01 (expt 4 (- k)))))))
    (match-define (list coef _) (at 0))
    (for ([k (in-list '(-3 5))])
      (law-case (format "~a ~a classifier without intercept, k ~a" dt loss k)
        (check-equal? (at k) (list (map (lambda (c) (* c (expt 2 (- k)))) coef) 0.0))))))

(test-gpu "y times 2^c with tol times 4^c gives coef and intercept times 2^c"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (define (at c)
      (reg dt
           rows
           (map (lambda (t) (* t (expt 2 c))) targets)
           (hash '#:penalty penalty '#:tol (* 1e-3 (expt 4 c)))))
    (match-define (list coef intercept) (at 0))
    (for ([c (in-list '(-3 5))])
      (define (scaled-by v)
        (* v (expt 2 c)))
      (law-case (format "~a ~a, c ~a" dt penalty c)
        (check-equal? (at c) (list (map scaled-by coef) (scaled-by intercept)))))))

(test-gpu "a batch size of n, n + 1 or the int32 maximum is one full batch"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (match-define (list n n+1 most)
        (for/list ([b (in-list '(300 301 2147483647))])
          (reg dt rows targets (hash '#:batch-size b))))
      (check-equal? n+1 n)
      (check-equal? most n))))

(test-gpu "an early stop is reached before 200 epochs for the squared and log losses"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a squared loss" dt)
      (check-equal? (reg dt rows targets (hash '#:epochs 200 '#:tol 1e-3))
                    (reg dt rows targets (hash '#:epochs 400 '#:tol 1e-3))))
    (for ([set (list (list "separable" blobs labels)
                     (list "overlapping" overlapping overlapping-labels))])
      (match-define (list name X y) set)
      (define (log-fit epochs)
        (clf-fit dt X y 'log (hash '#:epochs epochs '#:eta0 0.01)))
      (law-case (format "~a log ~a" dt name)
        (check-equal? (log-fit 200) (log-fit 400))))))

(test-gpu "the hinge and log intercepts are the coefficient of a column of ones"
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(hinge log))])
    (law-case (format "~a ~a" dt loss)
      (match-define (list coef intercept) (clf-fit dt blobs labels loss (hash '#:eta0 0.01)))
      (match-define (list ones-coef _)
        (clf-fit dt (with-ones blobs) labels loss (hash '#:eta0 0.01 '#:fit-intercept #f)))
      (check-equal? (append coef (list intercept)) ones-coef)
      (check-not-equal? intercept 0.0))))
