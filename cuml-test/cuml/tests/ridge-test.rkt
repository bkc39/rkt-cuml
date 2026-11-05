#lang racket/base

(require (only-in cuml fit predict r2-score ridge ridge-model ridge? score)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  linear-regression-result
                  linear-regression-result-coef
                  linear-regression-result-intercept
                  ridge-fit
                  ridge-predict
                  ridge-result-coef
                  ridge-result-intercept)
         (only-in racket/list cartesian-product)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/linear-data.rkt"
                  dtypes
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  normal-case
                  on-matrix
                  on-vector)
         (only-in "private/ridge-data.rkt" lstsq-algorithm ridge-case-result ridge-result-fit))

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

(define (outcome thunk coef-of intercept-of)
  (with-handlers ([exn:fail? (lambda (e) (regexp-replace #rx"^[a-z-]+: " (exn-message e) ""))])
    (define r (thunk))
    (list (~> r coef-of device-vector->list) (intercept-of r))))

(define (lstsq-outcome c dt solver fi)
  (outcome (lambda ()
             (linear-regression-fit (on-matrix (lcase-X c) dt)
                                    (on-vector (lcase-y c) dt)
                                    #:sample-weight (on-vector (lcase-w c) dt)
                                    #:algorithm (lstsq-algorithm solver)
                                    #:fit-intercept fi))
           linear-regression-result-coef
           linear-regression-result-intercept))

(test-gpu "alpha 0 is linear-regression-fit's fit, refusals included, bitwise (B1)"
  (for* ([c (in-list b1-cases)]
         [dt (in-list dtypes)]
         [solver (in-list '(auto eig svd))]
         [fi (in-list '(#t #f))]
         [alpha (in-list '(0 0.0 -0.0))])
    (test-case (format "~a ~a ~a ~a ~a" (lcase-name c) dt solver fi alpha)
      (check-equal?
       (outcome (lambda () (ridge-case-result c dt #:alpha alpha #:solver solver #:fit-intercept fi))
                ridge-result-coef
                ridge-result-intercept)
       (lstsq-outcome c dt solver fi)))))

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
        (ridge-result-fit (ridge-case-result c dt #:alpha 3/4 #:solver solver #:fit-intercept fi)))
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
