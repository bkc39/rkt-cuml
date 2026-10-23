#lang racket/base

(require (for-syntax racket/base)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define
         (only-in raft current-device-resources))

(provide deposit->native-model
         live-anchor
         native-model-kind
         native-model-release!
         native-model-released?
         native-model?
         with-native-models)

(struct native-model (kind anchor release state phantom) #:constructor-name anchor->native-model)

(define (deposit->native-model who kind d release #:opaque-bytes [opaque-bytes 0])
  (error 'unimplemented))

(define (live-anchor who m kind)
  (error 'unimplemented))

(define (native-model-released? m)
  (error 'unimplemented))

(define (native-model-release! m #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define-syntax-parse-rule (with-native-models (binding ...) body:expr ...+)
  (error 'unimplemented))
