#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre define-syntax-class expr id pattern))
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define-codes
         define-defaults)

(begin-for-syntax
  (define-syntax-class entry
    #:attributes (name value)
    (pattern [name:id value:expr])))

(define-syntax-parse-rule (define-defaults table:id e:entry ...)
  #:with default (format-id #'table "default")
  (begin
    (define table (hasheq (~@ 'e.name e.value) ...))
    (define (default key)
      (hash-ref table key))))

(define-syntax-parse-rule (define-codes code:id e:entry ...)
  (define code
    (let ([table (hasheq (~@ 'e.name e.value) ...)]) (lambda (key) (hash-ref table key -1)))))
