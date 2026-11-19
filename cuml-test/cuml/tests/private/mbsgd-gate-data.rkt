#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in "linear-data.rkt" law-case on-matrix on-vector)
         (only-in "mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "refusal-rows.rkt" check-row row))

(provide alternating
         blobs
         check-classifier-lower
         check-classifier-rise
         grid
         grid-rows
         grid-targets
         grid-y
         labels
         refused-at-eta0
         regressor-fit
         rose)

(define (rose eta0)
  (regexp (string-append "^the objective rose from [0-9.e+-]+ to [0-9.e+-]+; lower #:eta0 \\(now "
                         (regexp-quote eta0)
                         "\\)$")))

(define (printed eta0)
  (number->string (if (integer? eta0)
                      (inexact->exact eta0)
                      eta0)))

(define regressor-fit
  (make-keyword-procedure
   (lambda (kws vals X y dt)
     (lambda ()
       (keyword-apply mbsgd-regressor-fit kws vals (list (on-matrix X dt) (on-vector y dt)))))))

(define (refused-at-eta0 fit eta0)
  (check-row "mbsgd-regressor-fit"
             (row (format "eta0 ~a" (printed eta0)) (rose (printed eta0)) (fit eta0))))

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
