#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/match == match match-define)
         (only-in raft current-device-resources device-vector? dtype shape)
         (only-in raft/unsafe deposit->list)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define
         (only-in "define-who.rkt" define/who)
         (only-in "logic-error.rkt" logic-error)
         (only-in "release.rkt" with-release))

(provide deposit->native-model
         deposit->native-model/who
         live-anchor
         live-anchor/who
         native-model-kind
         native-model-release!
         native-model-released?
         native-model?
         with-native-models)

(struct native-model (kind [anchor #:mutable] release state phantom)
  #:constructor-name anchor->native-model
  #:property prop:custom-write
  (lambda (m port _mode)
    (fprintf port
             "#<native-model ~a~a>"
             (native-model-kind m)
             (if (native-model-released? m) " released" ""))))

(define phantoms (make-ephemeron-hasheq))

(define (anchor? a)
  (and (device-vector? a) (eq? (dtype a) 'int32) (equal? (shape a) '(1))))

(define/who (deposit->native-model who kind d release #:opaque-bytes [opaque-bytes 0])
  (match (deposit->list d)
    [(cons (? anchor? anchor) regions)
     (define phantom (make-phantom-bytes opaque-bytes))
     (for ([region (in-list regions)])
       (hash-set! phantoms region phantom))
     (values (anchor->native-model kind anchor release (box 'live) phantom) regions)]
    [_ (logic-error "~a: out: region 0 is not a model anchor" who)]))

(define (abandoned? state)
  (and (thread? state) (not (thread-running? state))))

(define (usable? state)
  (or (eq? state 'live) (abandoned? state)))

(define (native-model-released? m)
  (eq? (unbox (native-model-state m)) 'released))

(define (checked-model who m)
  (if (native-model? m)
      m
      (logic-error "~a: model: expected a native model, got ~e" who m)))

(define/who (live-anchor who m kind)
  (match m
    [(native-model (== kind) anchor _ (app unbox (? usable?)) _) anchor]
    [(native-model (== kind) _ _ _ _)
     (logic-error "~a: model: expected a live model, got a released one" who)]
    [(native-model other _ _ _ _)
     (logic-error "~a: model: expected a ~a model, got a ~a model" who kind other)]
    [_ (logic-error "~a: model: expected a native model, got ~e" who m)]))

(define (claim! state me)
  (match (unbox state)
    [(and seen (? usable?)) (or (box-cas! state seen me) (claim! state me))]
    [_ #f]))

(define (call-with-claim state release commit)
  (define me (current-thread))
  (parameterize-break #f
                      (when (claim! state me)
                        (dynamic-wind void
                                      (lambda ()
                                        (release)
                                        (set-box! state 'released)
                                        (commit))
                                      (lambda ()
                                        (when (eq? (unbox state) me)
                                          (set-box! state 'live))))))
  (parameterize-break (break-enabled) (void)))

(define (release-model who m resources)
  (match-define (native-model _ anchor release state phantom) m)
  (call-with-claim state
                   (lambda ()
                     (release who resources anchor)
                     (set-phantom-bytes! phantom 0))
                   (lambda () (set-native-model-anchor! m #f)))
  (when (native-model-released? m)
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
  (let ([m (checked-model 'with-native-models b.model)] ...)
    (with-release #:who 'with-native-models
                  ([b.name m (released-by 'with-native-models)] ...)
                  body ...)))
