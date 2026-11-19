#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in racket/list make-list)
         (only-in racket/match match)
         (only-in rackunit check-not-exn)
         (only-in "cuml-error.rkt" raised)
         (only-in "linear-data.rkt" law-case on-matrix on-vector)
         (only-in "mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "refusal-rows.rkt" check-row row))

(provide alternating
         blobs
         check-classifier-lower
         check-classifier-rise
         check-constant-batch-one
         constant-rows
         grid
         grid-rows
         grid-targets
         grid-y
         labels
         loss-r2
         lost-at-eta0
         off-constant-at-eta0
         off-predictions-at-eta0
         refused-at-eta0
         regressor-fit
         rose
         round-off-coef-steps
         round-off-intercept-share
         round-off-prediction-share
         wide-problem)

(define ((risen what) eta0)
  (regexp (string-append "^the "
                         what
                         " rose from [0-9.e+-]+ to [0-9.e+-]+; lower #:eta0 \\(now "
                         (regexp-quote eta0)
                         "\\)$")))

(define round-off-intercept-share 1e-6)
(define round-off-coef-steps 2)
(define round-off-prediction-share 0.0009765625)

(define off-constant
  #px"^y is constant to round-off and (a coefficient of the fit reaches|the fit's intercept is off by) [0-9.e+-]+; lower #:eta0 or #:alpha$")

(define off-predictions
  #px"^y is constant to round-off and the fit's predictions are off by [0-9.e+-]+; lower #:eta0 or #:alpha$")

(define rose (risen "objective"))
(define loss-rose (risen "loss"))

(define (printed eta0)
  (number->string (if (integer? eta0)
                      (inexact->exact eta0)
                      eta0)))

(define regressor-fit
  (make-keyword-procedure
   (lambda (kws vals X y dt)
     (lambda ()
       (keyword-apply mbsgd-regressor-fit kws vals (list (on-matrix X dt) (on-vector y dt)))))))

(define ((refused-by message) fit eta0)
  (check-row "mbsgd-regressor-fit"
             (row (format "eta0 ~a" (printed eta0)) (message (printed eta0)) (fit eta0))))

(define refused-at-eta0 (refused-by rose))
(define lost-at-eta0 (refused-by loss-rose))
(define off-constant-at-eta0 (refused-by (lambda (_) off-constant)))
(define off-predictions-at-eta0 (refused-by (lambda (_) off-predictions)))

(define (loss-r2 fit)
  (match (regexp-match #px"the loss rose from ([0-9.e+-]+) to ([0-9.e+-]+);"
                       (exn-message (raised fit)))
    [(list _ start end) (- 1 (/ (string->number end) (string->number start)))]))

(define constant-rows (normal-rows 300 4 3))

(define (check-constant-batch-one dt)
  (law-case (format "~a y = 1.7 l1 alpha 100 invscaling batch 1" dt)
    (off-constant-at-eta0 (lambda (eta0)
                            (regressor-fit constant-rows
                                           (make-list 300 1.7)
                                           dt
                                           #:penalty 'l1
                                           #:alpha 100
                                           #:eta0 eta0
                                           #:learning-rate 'invscaling
                                           #:batch-size 1))
                          1)))

(define (wide-problem n d)
  (define rows (normal-rows n d 11))
  (values rows
          (for/list ([r (in-list rows)]
                     [e (in-list (normal-rows n 1 12))])
            (+ (car r) (cadr r) (caddr r) (* 0.5 (car e))))))

(define (alternating count v)
  (for/list ([i (in-range count)])
    (if (even? i)
        v
        (- v))))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define grid-y '(1 2 3 4 5 6))

(define-values (blobs labels) (blob-rows 512 6 43))

(define (classifier-fit loss eta0)
  (lambda ()
    (mbsgd-classifier-fit (on-matrix blobs)
                          labels
                          #:loss loss
                          #:batch-size 1
                          #:alpha 10
                          #:eta0 eta0)))

(define (check-classifier-rise loss)
  (law-case (format "~a classifier refused at eta0 0.05" loss)
    (check-row "mbsgd-classifier-fit" (row "eta0 0.05" (rose "0.05") (classifier-fit loss 0.05)))))

(define (check-classifier-lower loss)
  (law-case (format "~a classifier passes at eta0 0.01" loss)
    (check-not-exn (classifier-fit loss 0.01))))

(define grid-rows (normal-rows 300 5 21))
(define grid-targets (reg-target grid-rows 21))
