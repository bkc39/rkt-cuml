#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/logistic-data.rkt" lr-case-X lr-case-y)
         (only-in "private/logistic-oracle.rkt" residual residual-of)
         (only-in "private/real-logistic-data.rkt" labelled-case labelled-labels labelled-set))

(define stationary-bound 1e-7)

(define (stamps span seed)
  (define normal (normal-source seed))
  (define uniform (uniform-source (add1 seed)))
  (for/lists (X y)
             ([_ (in-range 400)])
             (define a (normal))
             (define t (* span (uniform)))
             (values (list a (+ 1.7e9 t))
                     (if (positive? (+ a (* 4 (- (/ t span) 1/2)) (- (uniform) 1/2))) 1 0))))

(define (residual-at r X y penalty tol #:fit-intercept [fi #t])
  (residual-of X
               y
               #f
               (device-matrix->list* (logistic-regression-result-coef r))
               (device-vector->list (logistic-regression-result-intercept r))
               #:penalty penalty
               #:fit-intercept fi
               #:tol tol))

(test-gpu "epoch-second timestamps over a minute fit with an intercept, inside the check's bound"
  (for* ([seed (in-list '(5 6 7))]
         [penalty (in-list '(none l2))])
    (test-case (format "~a ~a" seed penalty)
      (define-values (X y) (stamps 60.0 seed))
      (define r
        (logistic-regression-fit (on-matrix X 'float64) (on-vector y 'int32) #:penalty penalty))
      (match-define (residual K _ bound) (residual-at r X y penalty 1e-4))
      (check-true (<= K bound) (format "K ~a bound ~a" (exact->inexact K) (exact->inexact bound))))))

(test-gpu "raw breast-cancer under 'none fits to the real-data stationarity bound"
  (define set (labelled-set 'breast-cancer "diagnosis" '("id")))
  (define X (lr-case-X (labelled-case set)))
  (define y (lr-case-y (labelled-case set)))
  (for ([arm (in-list '((float64 #f 1500) (float64 #t 1000) (float32 #t 1000)))])
    (match-define (list dt fi max-iter) arm)
    (test-case (format "~a ~a" dt fi)
      (define r
        (logistic-regression-fit (on-matrix X dt)
                                 (labelled-labels set)
                                 #:penalty 'none
                                 #:fit-intercept fi
                                 #:max-iter max-iter))
      (match-define (residual K F _) (residual-at r X y 'none 1e-4 #:fit-intercept fi))
      (check-true (<= K (* stationary-bound (max F 1)))
                  (format "K ~a F ~a" (exact->inexact K) (exact->inexact F))))))
