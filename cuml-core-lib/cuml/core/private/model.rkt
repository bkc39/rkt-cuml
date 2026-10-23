#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/match match)
         (only-in raft current-device-resources device-vector? dtype shape)
         (only-in raft/unsafe deposit->list)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define
         (only-in "../../core.rkt" exn:fail:cuml)
         (only-in "release.rkt" with-release))

(provide deposit->native-model
         live-anchor
         native-model-kind
         native-model-release!
         native-model-released?
         native-model?
         with-native-models)

(struct native-model (kind [anchor #:mutable] release state phantom)
  #:constructor-name anchor->native-model
  #:omit-define-syntaxes
  #:property prop:custom-write
  (lambda (m port _mode)
    (fprintf port
             "#<native-model ~a~a>"
             (native-model-kind m)
             (if (native-model-released? m) " released" ""))))

(define (refuse who message . args)
  (raise (exn:fail:cuml (format "~a: ~a" who (apply format message args))
                        (current-continuation-marks)
                        'logic)))

(define (anchor? a)
  (and (device-vector? a) (eq? (dtype a) 'int32) (equal? (shape a) '(1))))

(define (deposit->native-model who kind d release #:opaque-bytes [opaque-bytes 0])
  (match (deposit->list d)
    [(cons (? anchor? anchor) regions)
     (values (anchor->native-model kind anchor release (box 'live) (make-phantom-bytes opaque-bytes))
             regions)]
    [_ (refuse who "out: region 0 is not a model anchor")]))

(define (native-model-released? m)
  (eq? (unbox (native-model-state m)) 'released))

(define (checked-model who m)
  (if (native-model? m)
      m
      (refuse who "model: expected a native model, got ~e" m)))

(define (live-anchor who m kind)
  (define anchor (native-model-anchor (checked-model who m)))
  (cond
    [(not (eq? (native-model-kind m) kind))
     (refuse who "model: expected a ~a model, got a ~a model" kind (native-model-kind m))]
    [(and anchor (eq? (unbox (native-model-state m)) 'live)) anchor]
    [else (refuse who "model: expected a live model, got a released one")]))

(define (release-model who m resources)
  (define state (native-model-state m))
  (when (box-cas! state 'live 'releasing)
    (with-handlers ([(lambda (_) #t) (lambda (e)
                                       (set-box! state 'live)
                                       (raise e))])
      ((native-model-release m) who resources (native-model-anchor m)))
    (set-box! state 'released)
    (set-phantom-bytes! (native-model-phantom m) 0)
    (set-native-model-anchor! m #f)))

(define (native-model-release! m #:resources [resources (current-device-resources)])
  (release-model 'native-model-release! m resources))

(define ((released-by who) m)
  (release-model who m (current-device-resources)))

(begin-for-syntax
  (define-syntax-class model-binding
    #:description "a [name model-expr] binding"
    (pattern [name:id model:expr])))

(define-syntax-parse-rule (with-native-models (b:model-binding ...) body:expr ...+)
  #:fail-when (check-duplicate-identifier (syntax->list #'(b.name ...))) "duplicate binding name"
  #:with (m ...) (generate-temporaries #'(b.name ...))
  (let ([m b.model] ...)
    (with-release
     #:who 'with-native-models
     ([b.name (checked-model 'with-native-models m) (released-by 'with-native-models)] ...)
     body ...)))
