#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in racket/match match-define)
         (only-in rackunit check-exn check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/logistic-data.rkt" lr-case-X lr-case-y)
         (only-in "private/logistic-oracle.rkt" residual residual-of)
         (only-in "private/real-logistic-data.rkt" labelled-case labelled-labels labelled-set))

(define stationary-bound 1e-7)

(define raw-breast-cancer-arms '((float64 #f 6000) (float32 #t 6000)))

(define (stamps span seed)
  (define normal (normal-source seed))
  (define uniform (uniform-source (add1 seed)))
  (for/lists (X y #:result (values X y))
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

(test-gpu "a rescue that spends max-iter names its last attempt's residual and says raise max-iter"
  (define-values (X y) (stamps 60.0 5))
  (check-exn #rx"residual 0\\.0031397[0-9]* above the bound 0\\.0011177[0-9]* after 20 iterations \\(max-iter 20, tol 1e-04\\); raise max-iter$"
             (lambda ()
               (logistic-regression-fit (on-matrix X 'float64)
                                        (on-vector y 'int32)
                                        #:penalty 'l2
                                        #:max-iter 20))))

(test-gpu "raw breast-cancer under 'none fits to the real-data stationarity bound, given room"
  (define set (labelled-set 'breast-cancer "diagnosis" '("id")))
  (define X (lr-case-X (labelled-case set)))
  (define y (lr-case-y (labelled-case set)))
  (for ([arm (in-list raw-breast-cancer-arms)])
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
                  (format "K ~a F ~a" (exact->inexact K) (exact->inexact F)))
      (check-true (<= (logistic-regression-result-n-iter r) max-iter)
                  (format "n-iter ~a" (logistic-regression-result-n-iter r))))))

(test-gpu "the rescue's solves share max-iter: raw breast-cancer under 'none is refused at 1000"
  (define set (labelled-set 'breast-cancer "diagnosis" '("id")))
  (for ([dt (in-list '(float64 float32))])
    (test-case (format "~a" dt)
      (check-exn #rx"after 1000 iterations \\(max-iter 1000, tol 1e-04\\); raise max-iter$"
                 (lambda ()
                   (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) dt)
                                            (labelled-labels set)
                                            #:penalty 'none))))))
