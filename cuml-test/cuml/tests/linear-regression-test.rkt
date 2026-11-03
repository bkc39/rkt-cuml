#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-params
                  linear-regression-predict
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  algorithms
                  dtypes
                  lcase-X
                  lcase-w
                  lcase-y
                  normal-case
                  on-matrix
                  on-vector)
         (only-in "private/scaler-data.rkt" bits-equal?))

(define base (normal-case 200 5 0))
(define weighted (normal-case 200 5 1 #:weighted? #t))

(define (fit-of X y w dt #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (define r
    (linear-regression-fit (on-matrix X dt)
                           (on-vector y dt)
                           #:sample-weight (on-vector w dt)
                           #:algorithm a
                           #:fit-intercept fi))
  (list (device-vector->list (linear-regression-result-coef r))
        (linear-regression-result-intercept r)))

(define (case-fit c dt . options)
  (apply fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt options))

(define (match-fit c dt a fi proc)
  (apply proc (fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt #:algorithm a #:fit-intercept fi)))

(define (take2 l)
  (list (car l) (cadr l)))

(define (rounded dt x)
  (if (eq? dt 'float32)
      (real->double-flonum (real->single-flonum x))
      (real->double-flonum x)))

(define (scale-column rows k factor)
  (for/list ([row (in-list rows)])
    (for/list ([x (in-list row)] [j (in-naturals)])
      (if (= j k) (* x factor) x))))

(define (insert-column rows at value-of)
  (for/list ([row (in-list rows)] [i (in-naturals)])
    (append (for/list ([x (in-list row)] [j (in-range at)]) x)
            (list (value-of i))
            (list-tail row at))))

(define (remove-at xs at)
  (append (for/list ([x (in-list xs)] [j (in-range at)]) x) (list-tail xs (add1 at))))

(define (each-fit proc)
  (for* ([dt (in-list dtypes)] [a (in-list algorithms)] [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" dt a fi)
      (proc dt a fi))))

(test-gpu "S1, S3: coef has one element per column in X's dtype; the intercept is a flonum of that dtype"
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

(test-gpu "S2: predict returns one element per row in X's dtype"
  (for ([dt (in-list dtypes)])
    (define r (linear-regression-fit (on-matrix (lcase-X base) dt) (on-vector (lcase-y base) dt)))
    (for ([n (in-list '(1 1000))])
      (define X (on-matrix (for/list ([i (in-range n)]) (list-ref (lcase-X base) (modulo i 200))) dt))
      (define p (linear-regression-predict r X))
      (check-equal? (shape p) (list n))
      (check-equal? (dtype p) dt))))

(test-gpu "C1: eig is auto, svd-jacobi is svd, and auto is the default"
  (check-equal? (hash-ref (linear-regression-params) 'algorithm) 'auto)
  (for* ([dt (in-list dtypes)] [c (in-list (list base weighted))])
    (check-true (bits-equal? (case-fit c dt #:algorithm 'eig) (case-fit c dt #:algorithm 'auto)))
    (check-true (bits-equal? (case-fit c dt) (case-fit c dt #:algorithm 'auto)))
    (check-true (bits-equal? (case-fit c dt #:algorithm 'svd-jacobi) (case-fit c dt #:algorithm 'svd)))))

(define (powers dt)
  (if (eq? dt 'float32)
      '(20 -20 60 -60)
      '(500 -500)))

(test-gpu "E1: scaling a column by 2^m scales its coefficient by 2^-m and leaves the rest"
  (each-fit
   (lambda (dt a fi)
     (for ([c (in-list (list base weighted))])
       (match-fit c dt a fi
                  (lambda (coef b)
                    (for* ([m (in-list (powers dt))] [k (in-list '(0 3))])
                      (define scaled
                        (fit-of (scale-column (lcase-X c) k (expt 2 m)) (lcase-y c) (lcase-w c) dt
                                #:algorithm a #:fit-intercept fi))
                      (define expected
                        (for/list ([x (in-list coef)] [j (in-naturals)])
                          (if (= j k) (* x (expt 2.0 (- m))) x)))
                      (check-true (bits-equal? scaled (list expected b)) (format "m ~a k ~a" m k)))))))))

(test-gpu "E2: scaling y by 2^m scales coef and the intercept by 2^m"
  (each-fit
   (lambda (dt a fi)
     (for ([c (in-list (list base weighted))])
       (match-fit c dt a fi
                  (lambda (coef b)
                    (for ([m (in-list (powers dt))])
                      (define f (expt 2 m))
                      (check-true (bits-equal? (fit-of (lcase-X c) (map (lambda (v) (* v f)) (lcase-y c))
                                                       (lcase-w c) dt #:algorithm a #:fit-intercept fi)
                                               (list (map (lambda (x) (* x (exact->inexact f))) coef)
                                                     (* b (exact->inexact f))))
                                  (format "m ~a" m)))))))))

(test-gpu "E3: scaling the weights by 4^m leaves the fit unchanged"
  (each-fit
   (lambda (dt a fi)
     (define fitted (case-fit weighted dt #:algorithm a #:fit-intercept fi))
     (for ([m (in-list (if (eq? dt 'float32) '(30 -30) '(250 -250)))])
       (define w (map (lambda (v) (* v (expt 4 m))) (lcase-w weighted)))
       (check-true (bits-equal? (fit-of (lcase-X weighted) (lcase-y weighted) w dt
                                        #:algorithm a #:fit-intercept fi)
                                fitted)
                   (format "m ~a" m))))))

(test-gpu "E4: a constant (or, without an intercept, zero) column gets 0.0 and leaves the rest"
  (each-fit
   (lambda (dt a fi)
     (for ([c (in-list (list base weighted))])
       (match-fit c dt a fi
                  (lambda (coef b)
                    (for ([at (in-list '(0 2 5))])
                      (define X (insert-column (lcase-X c) at (lambda (_) (if fi 3/4 0))))
                      (match-define (list coef* b*)
                        (fit-of X (lcase-y c) (lcase-w c) dt #:algorithm a #:fit-intercept fi))
                      (check-equal? (list-ref coef* at) 0.0)
                      (check-true (bits-equal? (list (remove-at coef* at) b*) (list coef b))
                                  (format "at ~a" at))))))
     (define zero-rows '(3 17 40))
     (define w (for/list ([v (in-list (lcase-w weighted))] [i (in-naturals)])
                 (if (memv i zero-rows) 0 v)))
     (define X (insert-column (lcase-X weighted) 1 (lambda (i) (if (memv i zero-rows) 9 (if fi 5/2 0)))))
     (define dropped (fit-of (insert-column (lcase-X weighted) 1 (lambda (_) (if fi 5/2 0)))
                             (lcase-y weighted) w dt #:algorithm a #:fit-intercept fi))
     (check-true (bits-equal? (fit-of X (lcase-y weighted) w dt #:algorithm a #:fit-intercept fi) dropped)
                 "constant over the positive-weight rows only"))))

(test-gpu "E5: nothing to solve"
  (for* ([dt (in-list dtypes)] [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define X (lcase-X base))
      (check-equal? (fit-of X (make-list 200 13/4) #f dt #:algorithm a) (list (make-list 5 0.0) 3.25))
      (check-equal? (fit-of X (make-list 200 0) #f dt #:algorithm a #:fit-intercept #f)
                    (list (make-list 5 0.0) 0.0))
      (define one (for/list ([i (in-range 200)]) (if (= i 7) 3/2 0)))
      (check-equal? (fit-of X (lcase-y base) one dt #:algorithm a)
                    (list (make-list 5 0.0) (rounded dt (list-ref (lcase-y base) 7)))))))

(test-gpu "E6: fits and predictions repeat bitwise"
  (define big (normal-case 100000 7 3))
  (define big-weighted (normal-case 100000 7 4 #:weighted? #t))
  (define mid-weighted (normal-case 2000 7 5 #:weighted? #t))
  (for* ([dt (in-list dtypes)] [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (for ([c (in-list (list base big big-weighted mid-weighted))]
            [fi (in-list '(#t #t #f #t))])
        (define X (on-matrix (lcase-X c) dt))
        (define y (on-vector (lcase-y c) dt))
        (define w (on-vector (lcase-w c) dt))
        (define (once)
          (define r (linear-regression-fit X y #:sample-weight w #:algorithm a #:fit-intercept fi))
          (list (device-vector->list (linear-regression-result-coef r))
                (linear-regression-result-intercept r)
                r))
        (define first-fit (once))
        (for ([_ (in-range 4)])
          (check-true (bits-equal? (take2 (once)) (take2 first-fit))))
        (when (eq? c big)
          (define p (device-vector->list (linear-regression-predict (caddr first-fit) X)))
          (for ([_ (in-range 4)])
            (check-true (bits-equal? (device-vector->list (linear-regression-predict (caddr first-fit) X))
                                     p))))))))

(test-gpu "E7: the docs data, f64"
  (define X '((1 1) (1 2) (2 2) (2 3)))
  (define r (linear-regression-fit (on-matrix X 'float64) (on-vector '(6 8 9 11) 'float64)))
  (check-equal? (device-vector->list (linear-regression-predict r (on-matrix '((3 5) (2 5)) 'float64)))
                '(16.0 15.0))
  (for ([a (in-list algorithms)])
    (check-equal? (fit-of '((0 1) (1 1) (2 1) (3 1)) '(1 3 5 7) #f 'float64 #:algorithm a)
                  '((2.0 0.0) 1.0)
                  (format "~a" a))))
