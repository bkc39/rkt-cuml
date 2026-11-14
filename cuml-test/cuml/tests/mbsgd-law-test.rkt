#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in racket/list split-at)
         (only-in rackunit check-equal? check-not-equal?)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt"
                  blob-rows
                  clf-parts
                  dtype-penalties
                  normal-rows
                  reg-parts
                  reg-target
                  scaled
                  with-ones))

(define rows (normal-rows 300 5 21))
(define targets (reg-target rows 21))
(define-values (blobs labels) (blob-rows 300 4 22))
(define-values (overlapping overlapping-labels) (blob-rows 300 4 23 #:sep 0.5))

(define (reg dt X-rows y-rows . options)
  (reg-parts (apply mbsgd-regressor-fit
                    (on-matrix X-rows dt)
                    (on-vector y-rows dt)
                    #:batch-size 16
                    #:eta0 0.01
                    #:epochs 50
                    options)))

(define (clf dt X-rows ys . options)
  (clf-parts (apply mbsgd-classifier-fit (on-matrix X-rows dt) (on-vector ys 'int32) options)))

(test-gpu "without shuffle the random state changes nothing"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (define fits
        (for/list ([seed (in-list (list #f 0 7 4294967295))])
          (reg dt rows targets #:shuffle #f #:random-state seed)))
      (for ([f (in-list (cdr fits))])
        (check-equal? f (car fits))))))

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
           (scaled rows (expt 2 k))
           targets
           #:penalty penalty
           #:alpha (* 0.01 (scale-alpha penalty k))
           #:eta0 (* 0.01 (expt 4 (- k)))
           #:learning-rate lr
           #:fit-intercept intercept?))
    (define base (at 0))
    (for ([k (in-list '(-20 -3 5 30))])
      (law-case (format "~a ~a ~a fit-intercept ~a, k ~a" dt lr penalty intercept? k)
        (check-equal? (at k)
                      (list (map (lambda (c) (* c (expt 2 (- k)))) (car base)) (cadr base))))))
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(log hinge))])
    (define (at k)
      (clf dt
           (scaled blobs (expt 2 k))
           labels
           #:loss loss
           #:fit-intercept #f
           #:alpha (* 0.01 (expt 4 k))
           #:eta0 (* 0.01 (expt 4 (- k)))
           #:batch-size 16
           #:epochs 50))
    (define base (at 0))
    (for ([k (in-list '(-3 5))])
      (law-case (format "~a ~a classifier without intercept, k ~a" dt loss k)
        (check-equal? (at k) (list (map (lambda (c) (* c (expt 2 (- k)))) (car base)) 0.0))))))

(test-gpu "y times 2^c with tol times 4^c gives coef and intercept times 2^c"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (define (at c)
      (reg dt
           rows
           (map (lambda (t) (* t (expt 2 c))) targets)
           #:penalty penalty
           #:tol (* 1e-3 (expt 4 c))))
    (define base (at 0))
    (for ([c (in-list '(-3 5))])
      (law-case (format "~a ~a, c ~a" dt penalty c)
        (check-equal? (at c) (map-parts (lambda (v) (* v (expt 2 c))) base))))))

(define (map-parts f parts)
  (list (map f (car parts)) (f (cadr parts))))

(test-gpu "a batch size of n, n + 1 or the int32 maximum is one full batch"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (define fits
        (for/list ([b (in-list '(300 301 2147483647))])
          (reg dt rows targets #:batch-size b)))
      (check-equal? (cadr fits) (car fits))
      (check-equal? (caddr fits) (car fits)))))

(test-gpu "an early stop is reached before 200 epochs for the squared and log losses"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a squared loss" dt)
      (check-equal? (reg dt rows targets #:epochs 200 #:tol 1e-3)
                    (reg dt rows targets #:epochs 400 #:tol 1e-3)))
    (for ([set (list (cons blobs labels) (cons overlapping overlapping-labels))]
          [name (in-list '("separable" "overlapping"))])
      (law-case (format "~a log ~a" dt name)
        (check-equal? (clf dt (car set) (cdr set) #:loss 'log #:epochs 200 #:eta0 0.01)
                      (clf dt (car set) (cdr set) #:loss 'log #:epochs 400 #:eta0 0.01))))))

(test-gpu "the hinge and log intercepts are the coefficient of a column of ones"
  (for* ([dt (in-list dtypes)]
         [loss (in-list '(hinge log))])
    (law-case (format "~a ~a" dt loss)
      (define with (clf dt blobs labels #:loss loss #:eta0 0.01))
      (define-values (coef last)
        (split-at (car (clf dt (with-ones blobs) labels #:loss loss #:eta0 0.01 #:fit-intercept #f))
                  4))
      (check-equal? with (list coef (car last)))
      (check-not-equal? (cadr with) 0.0))))
