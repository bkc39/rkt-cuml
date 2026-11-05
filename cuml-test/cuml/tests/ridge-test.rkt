#lang racket/base

(require (only-in cuml fit predict r2-score ridge ridge-model ridge? score)
         (only-in cuml/linear-model
                  linear-regression-predict
                  linear-regression-result
                  ridge-fit
                  ridge-predict
                  ridge-result-coef
                  ridge-result-intercept)
         (only-in racket/list cartesian-product)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-pred check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/linear-data.rkt"
                  case-fit
                  dtypes
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  normal-case
                  on-matrix
                  on-vector)
         (only-in "private/ridge-data.rkt"
                  lstsq-algorithm
                  ridge-case-fit
                  ridge-case-result
                  ridge-result-fit
                  rounding-alpha))

(struct fitted (coef intercept) #:transparent)
(struct refused (message) #:transparent)

(define (outcome thunk)
  (with-handlers ([exn:fail? (lambda (e)
                               (refused (regexp-replace #rx"^[a-z-]+: " (exn-message e) "")))])
    (match-define (list coef b) (thunk))
    (fitted coef b)))

(define (collinear n)
  (define c (normal-case n 2 9))
  (lcase (format "collinear ~a" n)
         (for/list ([row (in-list (lcase-X c))])
           (append row (list (apply + row))))
         (lcase-y c)
         #f))

(define b1-cases
  (list (normal-case 200 7 0)
        (normal-case 200 7 1 #:weighted? #t)
        (normal-case 20 2 0)
        (normal-case 5000 7 0)
        (collinear 60)))

(define (same-as-lstsq c dt solver fi alpha)
  (check-equal?
   (outcome (lambda () (ridge-case-fit c dt #:alpha alpha #:solver solver #:fit-intercept fi)))
   (outcome (lambda () (case-fit c dt #:algorithm (lstsq-algorithm solver) #:fit-intercept fi)))))

(test-gpu "alpha 0 is linear-regression-fit's fit, refusals included, bitwise (B1)"
  (for* ([c (in-list b1-cases)]
         [dt (in-list dtypes)]
         [solver (in-list '(auto eig svd))]
         [fi (in-list '(#t #f))]
         [alpha (in-list '(0 0.0 -0.0))])
    (test-case (format "~a ~a ~a ~a ~a" (lcase-name c) dt solver fi alpha)
      (same-as-lstsq c dt solver fi alpha))))

(define threshold-cases
  (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t) (normal-case 20 2 0)))

(test-gpu "an alpha under rounding against the widest column spread is linear-regression-fit's fit (B1)"
  (for* ([c (in-list threshold-cases)]
         [dt (in-list dtypes)]
         [solver (in-list '(eig svd))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a ~a" (lcase-name c) dt solver fi)
      (same-as-lstsq c dt solver fi (/ (rounding-alpha c dt fi) 2)))))

(define tiny-alphas '(1e-40 1e-20 1e-10))

(define (collinear-outcome solver alpha)
  (outcome (lambda () (ridge-case-fit (collinear 60) 'float32 #:alpha alpha #:solver solver))))

(test-gpu "float32 collinear X with an alpha under rounding is linear-regression-fit's outcome"
  (for* ([solver (in-list '(eig svd))]
         [alpha (in-list tiny-alphas)])
    (test-case (format "~a ~a" solver alpha)
      (check-equal?
       (collinear-outcome solver alpha)
       (outcome (lambda ()
                  (case-fit (collinear 60) 'float32 #:algorithm (lstsq-algorithm solver))))))))

(test-gpu "float32 eig refuses collinear X with an alpha under rounding"
  (for ([alpha (in-list tiny-alphas)])
    (test-case (format "~a" alpha)
      (check-pred refused? (collinear-outcome 'eig alpha)))))

(test-gpu "two identical fits are bitwise equal (B7)"
  (for ([spec (in-list (cartesian-product (list (normal-case 200 7 1 #:weighted? #t)
                                                (normal-case 1000 7 2 #:weighted? #t)
                                                (normal-case 5000 7 3))
                                          dtypes
                                          '(eig svd)
                                          '(#t #f)))])
    (match-define (list c dt solver fi) spec)
    (test-case (format "~a ~a ~a ~a" (lcase-name c) dt solver fi)
      (define (once)
        (ridge-case-fit c dt #:alpha 3/4 #:solver solver #:fit-intercept fi))
      (check-equal? (once) (once)))))

(test-gpu "ridge-predict is linear-regression-predict on the same coefficients (B9)"
  (for ([dt (in-list dtypes)])
    (define c (normal-case 200 7 4))
    (define X (on-matrix (lcase-X c) dt))
    (define r (ridge-case-result c dt #:alpha 2))
    (check-equal? (device-vector->list (ridge-predict r X))
                  (~> (linear-regression-result (ridge-result-coef r) (ridge-result-intercept r))
                      (linear-regression-predict X)
                      device-vector->list))))

(test-gpu "the estimator's model, predict and score are ridge-fit's, bitwise (B10)"
  (define c (normal-case 200 7 5 #:weighted? #t))
  (for* ([dt (in-list dtypes)]
         [solver (in-list '(auto svd))])
    (test-case (format "~a ~a" dt solver)
      (define X (on-matrix (lcase-X c) dt))
      (define y (on-vector (lcase-y c) dt))
      (define w (on-vector (lcase-w c) dt))
      (define e (fit (ridge #:alpha 5/2 #:solver solver) X y #:sample-weight w))
      (define direct (ridge-fit X y #:sample-weight w #:alpha 5/2 #:solver solver))
      (check-true (ridge? e))
      (check-equal? (ridge-result-fit (ridge-model e)) (ridge-result-fit direct))
      (check-equal? (device-vector->list (predict e X))
                    (device-vector->list (ridge-predict direct X)))
      (check-equal? (score e X y #:sample-weight w)
                    (r2-score y (ridge-predict direct X) #:sample-weight w))))
  (check-false (ridge-model (ridge))))
