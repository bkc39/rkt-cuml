#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit logistic-regression-result-n-iter)
         (only-in racket/match match-define)
         (only-in rackunit check-exn check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" lr-case-X lr-case-y)
         (only-in "private/logistic-oracle.rkt" residual)
         (only-in "private/raw-breast-cancer.rkt" raw-breast-cancer)
         (only-in "private/real-logistic-data.rkt"
                  check-stationary
                  labelled-case
                  labelled-labels
                  residual-at))

(define raw-breast-cancer-arms '((float64 #f 6000) (float32 #t 6000) (float64 #t 20000)))

(define (stamps span seed)
  (define normal (normal-source seed))
  (define uniform (uniform-source (add1 seed)))
  (for/lists (X y #:result (values X y))
             ([_ (in-range 400)])
             (define a (normal))
             (define t (* span (uniform)))
             (values (list a (+ 1.7e9 t))
                     (if (positive? (+ a (* 4 (- (/ t span) 1/2)) (- (uniform) 1/2))) 1 0))))

(define (refusal fit)
  (with-handlers ([exn:fail? exn-message])
    (fit)
    "accepted"))

(test-gpu "epoch-second timestamps: fitted inside the check's bound"
  (for* ([seed (in-list '(5 6 7))]
         [penalty (in-list '(none l2))])
    (test-case (format "~a ~a" seed penalty)
      (define-values (X y) (stamps 60.0 seed))
      (define r
        (logistic-regression-fit (on-matrix X 'float64) (on-vector y 'int32) #:penalty penalty))
      (match-define (residual K _ bound) (residual-at r X y #:penalty penalty))
      (check-true (<= K bound) (format "K ~a bound ~a" (exact->inexact K) (exact->inexact bound))))))

(test-gpu "a rescue out of max-iter: names its last residual, says raise max-iter"
  (define-values (X y) (stamps 60.0 5))
  (match-define (regexp
                 #rx"residual ([^ ]+) above the bound ([^ ]+) after 20 iterations \\(max-iter 20, tol 1e-04\\); raise max-iter$"
                 (list _ K bound))
    (refusal (lambda ()
               (logistic-regression-fit (on-matrix X 'float64)
                                        (on-vector y 'int32)
                                        #:penalty 'l2
                                        #:max-iter 20))))
  (check-true (< (string->number bound) (string->number K) 0.05) (format "K ~a bound ~a" K bound)))

(test-gpu "raw 'none with room: stationary, within max-iter"
  (define X (lr-case-X (labelled-case raw-breast-cancer)))
  (define y (lr-case-y (labelled-case raw-breast-cancer)))
  (for ([arm (in-list raw-breast-cancer-arms)])
    (match-define (list dt fi max-iter) arm)
    (test-case (format "~a ~a" dt fi)
      (define r
        (logistic-regression-fit (on-matrix X dt)
                                 (labelled-labels raw-breast-cancer)
                                 #:penalty 'none
                                 #:fit-intercept fi
                                 #:max-iter max-iter))
      (check-stationary (residual-at r X y #:penalty 'none #:fit-intercept fi))
      (check-true (<= (logistic-regression-result-n-iter r) max-iter)
                  (format "n-iter ~a" (logistic-regression-result-n-iter r))))))

(test-gpu "raw 'none at the default max-iter: refused, the solves sharing it"
  (for ([dt (in-list '(float64 float32))])
    (test-case (format "~a" dt)
      (check-exn #rx"after 1000 iterations \\(max-iter 1000, tol 1e-04\\); raise max-iter$"
                 (lambda ()
                   (logistic-regression-fit (on-matrix (lr-case-X (labelled-case raw-breast-cancer))
                                                       dt)
                                            (labelled-labels raw-breast-cancer)
                                            #:penalty 'none))))))

(test-gpu "raw 'none at max-iter 6000: refused, saying raise it"
  (check-exn #rx"after 6000 iterations \\(max-iter 6000, tol 1e-04\\); raise max-iter$"
             (lambda ()
               (logistic-regression-fit (on-matrix (lr-case-X (labelled-case raw-breast-cancer))
                                                   'float64)
                                        (labelled-labels raw-breast-cancer)
                                        #:penalty 'none
                                        #:max-iter 6000))))

(test-gpu "raw 'l2 without an intercept: refused without a second solve"
  (define X (on-matrix (lr-case-X (labelled-case raw-breast-cancer)) 'float64))
  (define (iterations fi)
    (match-define (regexp
                   #rx"after ([0-9]+) iterations \\(max-iter 4000, tol 1e-04\\); standardise the columns$"
                   (list _ n))
      (refusal (lambda ()
                 (logistic-regression-fit X
                                          (labelled-labels raw-breast-cancer)
                                          #:penalty 'l2
                                          #:fit-intercept fi
                                          #:max-iter 4000))))
    (string->number n))
  (define alone (iterations #f))
  (define rescued (iterations #t))
  (check-true (< alone 4000) (format "n-iter ~a" alone))
  (check-true (> rescued alone) (format "n-iter ~a with an intercept, ~a without" rescued alone)))
