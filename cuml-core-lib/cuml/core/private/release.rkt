#lang racket/base

(require (for-syntax racket/base)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define)

(provide call-with-release
         with-release)

(define (call-with-release who acquired release proc)
  (error 'unimplemented))

(define-syntax-parse-rule (with-release (~optional (~seq #:who who:expr)) binding ... body:expr ...+)
  (error 'unimplemented))
