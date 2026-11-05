#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-params
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/list append* make-list take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/linear-data.rkt"
                  algorithms
                  case-fit
                  check-law
                  dtypes
                  fit-gap
                  fit-of
                  kept-rows
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  normal-case
                  on-matrix
                  on-vector
                  power-spread-case
                  power-spread-gone
                  result-fit
                  rounded
                  unit-roundoff)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define base (normal-case 200 5 0))
(define weighted (normal-case 200 5 1 #:weighted? #t))

(define (scale-column rows k factor)
  (for/list ([row (in-list rows)])
    (for/list ([x (in-list row)]
               [j (in-naturals)])
      (if (= j k)
          (* x factor)
          x))))

(define (insert-column rows at value-of)
  (for/list ([row (in-list rows)]
             [i (in-naturals)])
    (append (for/list ([x (in-list row)]
                       [_ (in-range at)])
              x)
            (list (value-of i))
            (list-tail row at))))

(define (remove-at xs at)
  (append (for/list ([x (in-list xs)]
                     [_ (in-range at)])
            x)
          (list-tail xs (add1 at))))

(define (each-fit proc)
  (for* ([dt (in-list dtypes)]
         [a (in-list algorithms)]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" dt a fi)
      (proc dt a fi))))

(test-gpu "coef has one element per column in X's dtype; the intercept is a flonum of that dtype"
  (each-fit (lambda (dt a fi)
              (define r
                (linear-regression-fit (on-matrix (lcase-X base) dt)
                                       (on-vector (lcase-y base) dt)
                                       #:algorithm a
                                       #:fit-intercept fi))
              (define coef (linear-regression-result-coef r))
              (define b (linear-regression-result-intercept r))
              (check-equal? (shape coef) '(5))
              (check-equal? (dtype coef) dt)
              (check-true (flonum? b))
              (check-equal? b (rounded dt b))
              (unless fi
                (check-true (eqv? b 0.0))))))

(test-gpu "predict returns one element per row in X's dtype"
  (for ([dt (in-list dtypes)])
    (define r (linear-regression-fit (on-matrix (lcase-X base) dt) (on-vector (lcase-y base) dt)))
    (for ([X (in-list (list (on-matrix (take (lcase-X base) 1) dt)
                            (on-matrix (append* (make-list 5 (lcase-X base))) dt)))]
          [n (in-list '(1 1000))])
      (define p (linear-regression-predict r X))
      (check-equal? (shape p) (list n))
      (check-equal? (dtype p) dt))))

(test-gpu "eig is auto, svd-jacobi is svd, and auto is the default"
  (check-equal? (hash-ref (linear-regression-params) 'algorithm) 'auto)
  (for* ([dt (in-list dtypes)]
         [c (in-list (list base weighted))])
    (check-true (bits-equal? (case-fit c dt #:algorithm 'eig) (case-fit c dt #:algorithm 'auto)))
    (check-true (bits-equal? (case-fit c dt) (case-fit c dt #:algorithm 'auto)))
    (check-true (bits-equal? (case-fit c dt #:algorithm 'svd-jacobi)
                             (case-fit c dt #:algorithm 'svd)))))

(define (powers dt)
  (if (eq? dt 'float32)
      '(20 -20 60 -60)
      '(500 -500)))

(test-gpu "scaling a column by 2^m scales its coefficient by 2^-m and leaves the rest"
  (each-fit (lambda (dt a fi)
              (for* ([c (in-list (list base weighted))]
                     #:do [(match-define (list coef b)
                             (case-fit c dt #:algorithm a #:fit-intercept fi))]
                     [m (in-list (powers dt))]
                     [k (in-list '(0 3))])
                (define scaled
                  (fit-of (scale-column (lcase-X c) k (expt 2 m))
                          (lcase-y c)
                          (lcase-w c)
                          dt
                          #:algorithm a
                          #:fit-intercept fi))
                (define expected
                  (for/list ([x (in-list coef)]
                             [j (in-naturals)])
                    (if (= j k)
                        (* x (expt 2.0 (- m)))
                        x)))
                (check-true (bits-equal? scaled (list expected b)) (format "m ~a k ~a" m k))))))

(test-gpu "scaling y by 2^m scales coef and the intercept by 2^m"
  (each-fit (lambda (dt a fi)
              (for* ([c (in-list (list base weighted))]
                     #:do [(match-define (list coef b)
                             (case-fit c dt #:algorithm a #:fit-intercept fi))]
                     [m (in-list (powers dt))])
                (define f (expt 2 m))
                (check-true (bits-equal? (fit-of (lcase-X c)
                                                 (map (lambda (v) (* v f)) (lcase-y c))
                                                 (lcase-w c)
                                                 dt
                                                 #:algorithm a
                                                 #:fit-intercept fi)
                                         (list (map (lambda (x) (* x (exact->inexact f))) coef)
                                               (* b (exact->inexact f))))
                            (format "m ~a" m))))))

(test-gpu "scaling the weights by 4^m leaves the fit unchanged"
  (each-fit (lambda (dt a fi)
              (define fitted (case-fit weighted dt #:algorithm a #:fit-intercept fi))
              (for ([m (in-list (if (eq? dt 'float32)
                                    '(30 -30)
                                    '(250 -250)))])
                (define w (map (lambda (v) (* v (expt 4 m))) (lcase-w weighted)))
                (check-true
                 (bits-equal?
                  (fit-of (lcase-X weighted) (lcase-y weighted) w dt #:algorithm a #:fit-intercept fi)
                  fitted)
                 (format "m ~a" m))))))

(test-gpu "a constant (or, without an intercept, zero) column gets 0.0 and leaves the rest"
  (each-fit (lambda (dt a fi)
              (for* ([c (in-list (list base weighted))]
                     #:do [(define fitted (case-fit c dt #:algorithm a #:fit-intercept fi))]
                     [at (in-list '(0 2 5))])
                (define X (insert-column (lcase-X c) at (lambda (_) (if fi 3/4 0))))
                (match-define (list coef* b*)
                  (fit-of X (lcase-y c) (lcase-w c) dt #:algorithm a #:fit-intercept fi))
                (check-equal? (list-ref coef* at) 0.0)
                (check-true (bits-equal? (list (remove-at coef* at) b*) fitted) (format "at ~a" at)))
              (define zero-rows '(3 17 40))
              (define w
                (for/list ([v (in-list (lcase-w weighted))]
                           [i (in-naturals)])
                  (if (memv i zero-rows) 0 v)))
              (define X
                (insert-column (lcase-X weighted)
                               1
                               (lambda (i)
                                 (cond
                                   [(memv i zero-rows) 9]
                                   [fi 5/2]
                                   [else 0]))))
              (define dropped
                (fit-of (insert-column (lcase-X weighted) 1 (lambda (_) (if fi 5/2 0)))
                        (lcase-y weighted)
                        w
                        dt
                        #:algorithm a
                        #:fit-intercept fi))
              (check-true
               (bits-equal? (fit-of X (lcase-y weighted) w dt #:algorithm a #:fit-intercept fi)
                            dropped)
               "constant over the positive-weight rows only"))))

(test-gpu "E4 holds bitwise when a column's weighted spread is within ulps of a power of two"
  (for* ([step (in-range -10 26 2)]
         #:do [(define c (power-spread-case 1 step))
               (define (kept xs)
                 (kept-rows xs power-spread-gone))
               (define X (kept (lcase-X c)))
               (define y (kept (lcase-y c)))
               (define w (kept (lcase-w c)))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" (lcase-name c) a)
      (match-define (list coef* b*)
        (fit-of (insert-column X 0 (lambda (_) 7.0)) y w 'float64 #:algorithm a))
      (check-equal? (car coef*) 0.0)
      (check-true (bits-equal? (list (remove-at coef* 0) b*) (fit-of X y w 'float64 #:algorithm a))
                  "E4: a constant column leaves the rest"))))

(test-gpu "nothing to solve: a constant y, a zero y, or a single positive weight"
  (for* ([dt (in-list dtypes)]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define X (lcase-X base))
      (check-equal? (fit-of X (make-list 200 13/4) #f dt #:algorithm a) (list (make-list 5 0.0) 3.25))
      (check-equal? (fit-of X (make-list 200 0) #f dt #:algorithm a #:fit-intercept #f)
                    (list (make-list 5 0.0) 0.0))
      (define one
        (for/list ([i (in-range 200)])
          (if (= i 7) 3/2 0)))
      (check-equal? (fit-of X (lcase-y base) one dt #:algorithm a)
                    (list (make-list 5 0.0) (rounded dt (list-ref (lcase-y base) 7)))))))

(test-gpu "with every column constant the intercept is the weighted mean of y"
  (for* ([dt (in-list dtypes)]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define X '((2 7) (2 7) (2 7) (2 7) (2 7)))
      (define y '(1 2 3 4 6))
      (check-equal? (fit-of X y #f dt #:algorithm a) (list '(0.0 0.0) (rounded dt 16/5)))
      (check-equal? (fit-of X y '(1 0 0 2 3) dt #:algorithm a) (list '(0.0 0.0) (rounded dt 9/2))))))

(test-gpu "fits and predictions repeat bitwise"
  (define big (normal-case 100000 7 3))
  (define big-weighted (normal-case 100000 7 4 #:weighted? #t))
  (define mid-weighted (normal-case 2000 7 5 #:weighted? #t))
  (for* ([dt (in-list dtypes)]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (for ([c (in-list (list base big big-weighted mid-weighted))]
            [fi (in-list '(#t #t #f #t))])
        (define X (on-matrix (lcase-X c) dt))
        (define y (on-vector (lcase-y c) dt))
        (define w (on-vector (lcase-w c) dt))
        (define (once)
          (linear-regression-fit X y #:sample-weight w #:algorithm a #:fit-intercept fi))
        (define r (once))
        (define fitted (result-fit r))
        (for ([_ (in-range 4)])
          (check-true (bits-equal? (result-fit (once)) fitted)))
        (when (eq? c big)
          (define (predicted)
            (~> r (linear-regression-predict X) device-vector->list))
          (define p (predicted))
          (for ([_ (in-range 4)])
            (check-true (bits-equal? (predicted) p))))))))

(test-gpu "the docs data fits under every algorithm, in float64"
  (define X '((1 1) (1 2) (2 2) (2 3)))
  (define y '(6 8 9 11))
  (define r (linear-regression-fit (on-matrix X) (on-vector y)))
  (check-equal? (~> r (linear-regression-predict (on-matrix '((3 5) (2 5)))) device-vector->list)
                '(16.0 15.0))
  (for ([a (in-list algorithms)])
    (check-law (format "~a" a)
               (fit-gap (fit-of X y #f 'float64 #:algorithm a) '((1 2) 3))
               (* 64 (unit-roundoff 'float64)))
    (check-equal? (fit-of '((0 1) (1 1) (2 1) (3 1)) '(1 3 5 7) #f 'float64 #:algorithm a)
                  '((2.0 0.0) 1.0)
                  (format "~a" a))))
