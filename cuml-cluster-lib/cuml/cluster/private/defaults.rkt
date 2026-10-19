#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module: its syntax classes come with it
                     syntax/parse/pre)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define-defaults)

(define-syntax-parse-rule (define-defaults table:id accessor:id [name:id value:expr] ...)
  (begin
    (define table (hasheq (~@ 'name value) ...))
    (define (accessor key)
      (hash-ref table key))))
