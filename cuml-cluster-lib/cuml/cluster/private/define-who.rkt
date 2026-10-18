#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     ;; whole-module: its syntax classes come with it
                     syntax/parse/pre)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define/who)

(begin-for-syntax
  (define-splicing-syntax-class formal
                                #:attributes ([spec 1] [argument 1])
                                (pattern id:id
                                  #:with (spec ...) #'(id)
                                  #:with (argument ...) #'(id))
                                (pattern (~seq kw:keyword [id:id default:expr])
                                  #:with (spec ...) #'(kw [id default])
                                  #:with (argument ...) #'(kw id))))

(define-syntax-parse-rule (define/who (name:id who:id f:formal ...)
                            body:expr ...+)
  #:with name/who (format-id #'name "~a/who" #'name #:source #'name)
  (begin
    (define (name/who who f.spec ... ...)
      body ...)
    (define (name f.spec ... ...)
      (name/who 'name f.argument ... ...))))
