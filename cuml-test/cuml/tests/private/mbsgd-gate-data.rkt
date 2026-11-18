#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit)
         (only-in rackunit check-not-exn)
         (only-in "linear-data.rkt" law-case on-matrix)
         (only-in "mbsgd-data.rkt" blob-rows normal-rows reg-target)
         (only-in "refusal-rows.rkt" check-row row))

(provide blobs
         check-classifier-lower
         check-classifier-rise
         grid-rows
         grid-targets
         labels
         rose)

(define (rose eta0)
  (regexp (string-append "^the objective rose from [0-9.e+-]+ to [0-9.e+-]+; lower #:eta0 \\(now "
                         (regexp-quote eta0)
                         "\\)$")))

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
