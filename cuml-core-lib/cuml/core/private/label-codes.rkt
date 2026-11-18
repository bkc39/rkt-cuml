#lang racket/base

(provide label-codes/who
         prop:label-codes)

(define-values (prop:label-codes _label-codes? label-codes-ref)
  (make-struct-type-property 'label-codes))

(define (label-codes/who who result y resources)
  ((label-codes-ref result) who result y resources))
