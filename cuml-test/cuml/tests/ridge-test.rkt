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
         (only-in rackunit
                  check-equal?
                  check-false
                  check-pred
                  check-regexp-match
                  check-true
                  test-case)
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

(struct refused (message) #:transparent)

(define (outcome thunk)
  (with-handlers ([exn:fail? (lambda (e)
                               (refused (regexp-replace #rx"^[a-z-]+: " (exn-message e) "")))])
    (thunk)))

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

(define (columns-apart lo hi)
  (define c (normal-case 60 2 3))
  (lcase (format "columns at 2^~a and 2^~a" lo hi)
         (for/list ([row (in-list (lcase-X c))])
           (match-define (list a b) row)
           (list (* a (expt 2 lo)) (* b (expt 2 hi))))
         (for/list ([row (in-list (lcase-X c))]
                    [y (in-list (lcase-y c))])
           (match-define (list a b) row)
           (+ a (* 1/2 b) (* 1/16 (- y 3))))
         #f))

(define subnormal-weight
  (let ([c (normal-case 40 1 4)])
    (lcase "a column varying only on a row of weight 2^-1074"
           (for/list ([row (in-list (lcase-X c))]
                      [i (in-naturals)])
             (list (if (= i 39) 3/4 1) (car row)))
           (lcase-y c)
           (for/list ([i (in-range 40)])
             (if (= i 39)
                 (expt 2 -1074)
                 1)))))

(test-gpu "alpha 0 is linear-regression-fit's fit on float64 spreads that underflow when staged (B1)"
  (for* ([c (in-list (list (columns-apart -1000 1000) (columns-apart -950 950) subnormal-weight))]
         [solver (in-list '(auto eig svd))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" (lcase-name c) solver fi)
      (same-as-lstsq c 'float64 solver fi 0))))

(define (kept-one)
  (define c (normal-case 30 1 3 #:weighted? #t))
  (lcase "one of three columns kept"
         (for/list ([row (in-list (lcase-X c))])
           (list 2 (car row) 0))
         (lcase-y c)
         (lcase-w c)))

(define (weights-only-at c rows)
  (lcase (format "~a, weights only at rows ~a" (lcase-name c) rows)
         (lcase-X c)
         (lcase-y c)
         (for/list ([i (in-range (length (lcase-y c)))])
           (if (memv i rows) 1 0))))

(define shape-cases
  (list (normal-case 5 8 4)
        (normal-case 6 8 5 #:weighted? #t)
        (kept-one)
        (weights-only-at (normal-case 10 3 2) '(4))
        (weights-only-at (normal-case 10 3 2) '(0 3))
        (normal-case 3 3 1)))

(test-gpu "alpha 0 or under rounding is linear-regression-fit's fit on m < d, one kept column and one positive row (B1)"
  (for* ([c (in-list shape-cases)]
         [dt (in-list dtypes)]
         [solver (in-list '(auto eig svd))]
         [fi (in-list '(#t #f))]
         [alpha (in-list '(0 1e-300))])
    (test-case (format "~a ~a ~a ~a ~a" (lcase-name c) dt solver fi alpha)
      (same-as-lstsq c dt solver fi alpha))))

(define threshold-cases
  (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t) (normal-case 20 2 0)))

(test-gpu "an alpha under rounding against the narrowest column spread is linear-regression-fit's fit (B1)"
  (for* ([c (in-list threshold-cases)]
         [dt (in-list dtypes)]
         [solver (in-list '(eig svd))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a ~a" (lcase-name c) dt solver fi)
      (same-as-lstsq c dt solver fi (/ (rounding-alpha c dt fi) 2)))))

(define tiny-alphas '(1e-40 1e-20 1e-10))

(define (collinear-outcome solver alpha)
  (outcome (lambda () (ridge-case-fit (collinear 60) 'float32 #:alpha alpha #:solver solver))))

(test-gpu "float32 collinear X with an alpha under rounding is refused, as linear-regression-fit refuses it"
  (for* ([solver (in-list '(eig svd))]
         [alpha (in-list tiny-alphas)])
    (test-case (format "~a ~a" solver alpha)
      (define ours (collinear-outcome solver alpha))
      (check-pred refused? ours)
      (check-regexp-match #rx"X may have collinear columns$" (refused-message ours))
      (check-equal?
       ours
       (outcome (lambda ()
                  (case-fit (collinear 60) 'float32 #:algorithm (lstsq-algorithm solver))))))))

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
