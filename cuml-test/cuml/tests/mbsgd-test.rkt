#lang racket/base

(require (only-in cuml fit mbsgd-classifier mbsgd-regressor predict score)
         (only-in cuml/linear-model
                  mbsgd-classifier-decision-function
                  mbsgd-classifier-fit
                  mbsgd-classifier-predict
                  mbsgd-classifier-result-classes
                  mbsgd-classifier-result-coef
                  mbsgd-classifier-result-intercept
                  mbsgd-classifier-result-loss
                  mbsgd-regressor-fit
                  mbsgd-regressor-predict
                  mbsgd-regressor-result-coef
                  mbsgd-regressor-result-intercept)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in raft device-matrix->list* device-vector->list device-vector? dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" decimals dtypes on-matrix)
         (only-in "private/mbsgd-data.rkt" blob-rows normal-rows reg-target))

(define rows (normal-rows 40 3 11))
(define targets (reg-target rows 11))
(define-values (blobs labels) (blob-rows 40 3 12))

(test-gpu "fits have one coefficient per column in X's dtype and predictions one value per row"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))])
    (test-case (format "~a, fit-intercept ~a" dt intercept?)
      (define X (on-matrix rows dt))
      (define r (mbsgd-regressor-fit X (on-vector targets dt) #:fit-intercept intercept?))
      (check-equal? (shape (mbsgd-regressor-result-coef r)) '(3))
      (check-equal? (dtype (mbsgd-regressor-result-coef r)) dt)
      (check-pred flonum? (mbsgd-regressor-result-intercept r))
      (unless intercept?
        (check-equal? (mbsgd-regressor-result-intercept r) 0.0))
      (check-equal? (shape (mbsgd-regressor-predict r X)) '(40))
      (check-equal? (shape (mbsgd-regressor-predict r (on-matrix '((1 2 3)) dt))) '(1))
      (for ([loss (in-list '(hinge log squared-loss))])
        (define c
          (mbsgd-classifier-fit (on-matrix blobs dt)
                                (on-vector labels 'int32)
                                #:loss loss
                                #:fit-intercept intercept?))
        (check-equal? (dtype (mbsgd-classifier-result-coef c)) dt)
        (check-equal? (shape (mbsgd-classifier-result-coef c)) '(3))
        (check-pred flonum? (mbsgd-classifier-result-intercept c))
        (check-equal? (mbsgd-classifier-result-loss c) loss)
        (check-equal? (mbsgd-classifier-result-classes c) '(0 1))
        (check-equal? (shape (mbsgd-classifier-decision-function c X)) '(40))
        (define predicted (mbsgd-classifier-predict c X))
        (check-true (and (device-vector? predicted) (eq? (dtype predicted) 'int32)))
        (check-equal? (shape (mbsgd-classifier-predict c (on-matrix '((9 9 9)) dt))) '(1))
        (define listed
          (mbsgd-classifier-predict
           (mbsgd-classifier-fit (on-matrix blobs dt) (map add1 labels) #:loss loss)
           X))
        (check-true (and (list? listed)
                         (= (length listed) 40)
                         (andmap (lambda (l) (and (memv l '(1 2)) #t)) listed)))))))

(test-gpu "a fit leaves the caller's X and y bit for bit as they were"
  (for* ([dt (in-list dtypes)]
         [intercept? (in-list '(#t #f))]
         [shuffle? (in-list '(#t #f))])
    (test-case (format "~a, fit-intercept ~a, shuffle ~a" dt intercept? shuffle?)
      (define X (on-matrix rows dt))
      (define y (on-vector targets dt))
      (define before (list (device-matrix->list* X) (device-vector->list y)))
      (mbsgd-regressor-fit X y #:fit-intercept intercept? #:shuffle shuffle? #:random-state 3)
      (check-equal? (list (device-matrix->list* X) (device-vector->list y)) before)
      (define Xc (on-matrix blobs dt))
      (define yc (on-vector labels dt))
      (define before-c (list (device-matrix->list* Xc) (device-vector->list yc)))
      (for ([loss (in-list '(hinge log squared-loss))])
        (mbsgd-classifier-fit Xc yc #:loss loss #:fit-intercept intercept? #:shuffle shuffle?))
      (check-equal? (list (device-matrix->list* Xc) (device-vector->list yc)) before-c))))

(test-gpu "the manual's examples print what it shows"
  (define X (on-matrix '((0 1) (1 0) (2 1) (3 0) (4 1))))
  (define y (on-vector '(1.0 2.0 4.0 5.0 7.0)))
  (define fitted (mbsgd-regressor-fit X y #:eta0 0.05 #:batch-size 5 #:shuffle #f))
  (check-equal? (decimals 3 (device-vector->list (mbsgd-regressor-result-coef fitted)))
                '("1.498" "0.253"))
  (define Xc (on-matrix '((0 0) (1 0) (0 1) (3 3) (4 3) (3 4))))
  (define fitted-c (mbsgd-classifier-fit Xc '(a a a b b b) #:eta0 0.05 #:batch-size 6))
  (check-equal? (mbsgd-classifier-result-classes fitted-c) '(a b))
  (check-equal? (mbsgd-classifier-predict fitted-c (on-matrix '((0 0) (4 4)))) '(a b))
  (check-equal?
   (real->decimal-string (score (fit (mbsgd-regressor #:eta0 0.05 #:batch-size 5) X y) X y) 3)
   "0.997")
  (check-equal? (predict (fit (mbsgd-classifier #:eta0 0.05 #:batch-size 6) Xc '(a a a b b b)) Xc)
                '(a a a b b b)))
