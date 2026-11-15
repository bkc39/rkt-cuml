#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre expr id))
         (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match match-define struct*)
         (only-in raft
                  current-device-resources
                  device-matrix?
                  device-vector->list
                  device-vector?
                  dtype)
         (only-in raft/unsafe deposit->list)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "glm.rkt"
                  bounded-count/who
                  checked-coef/who
                  flag/who
                  glm-predict/who
                  intercept-flag/who
                  nonnegative-value/who
                  penalty-code
                  positive-value/who)
         (only-in "labels.rkt"
                  class-view
                  decode-labels
                  device-kind?
                  fit-labels/who
                  label-codes/who
                  label-kind?
                  list-kind?
                  y-kind/who)
         (only-in "native.rkt"
                  cuml-linear-model-mbsgd-classifier-fit-job
                  cuml-linear-model-mbsgd-regressor-fit-job
                  cuml-linear-model-threshold-labels-job))

(provide (struct-out mbsgd-classifier-result)
         (struct-out mbsgd-regressor-result)
         mbsgd-classifier-decision-function
         mbsgd-classifier-decision-function/who
         mbsgd-classifier-fit
         mbsgd-classifier-fit/who
         mbsgd-classifier-params
         mbsgd-classifier-predict
         mbsgd-classifier-predict/who
         mbsgd-regressor-fit
         mbsgd-regressor-fit/who
         mbsgd-regressor-params
         mbsgd-regressor-predict
         mbsgd-regressor-predict/who)

(struct mbsgd-regressor-result (coef intercept) #:transparent)

(struct mbsgd-classifier-result (coef intercept classes label-kind loss)
  #:guard
  (lambda (coef intercept classes kind loss name)
    (cond
      [(not (device-vector? coef))
       (logic-error "~a: coef: expected a device-vector, got ~e" name coef)]
      [(not (flonum? intercept))
       (logic-error "~a: intercept: expected a flonum, got ~e" name intercept)]
      [(not (and (list? classes) (= (length classes) 2)))
       (logic-error "~a: classes: expected a list of 2 labels, got ~e" name classes)]
      [(not (label-kind? kind))
       (logic-error "~a: label-kind: expected the label kind of a fitted result, got ~e" name kind)]
      [(not (memq loss '(squared-loss log hinge)))
       (logic-error "~a: loss: expected one of 'hinge 'log 'squared-loss, got ~e" name loss)]
      [else (values coef intercept classes kind loss)]))
  #:property prop:label-codes
  (lambda (who result y resources)
    (match-define (struct* mbsgd-classifier-result ([classes classes] [label-kind kind])) result)
    (label-codes/who who classes kind y resources)))

(define-syntax-parse-rule (define-sgd-params name:id loss-default:expr)
  (define-hyperparameters name
    [loss loss-default]
    [penalty 'l2]
    [alpha 1e-4]
    [l1-ratio 0.15]
    [fit-intercept #t]
    [epochs 1000]
    [tol 1e-3]
    [shuffle #t]
    [learning-rate 'constant]
    [eta0 1e-3]
    [power-t 0.5]
    [batch-size 32]
    [n-iter-no-change 5]
    [random-state #f]))

(define-sgd-params mbsgd-regressor-params 'squared-loss)
(define-sgd-params mbsgd-classifier-params 'hinge)

(define-codes loss-code [squared-loss 0] [log 1] [hinge 2])
(define-codes learning-rate-code [constant 1] [invscaling 2] [adaptive 3])

(define seed-most (sub1 (expt 2 32)))

(define (coded who key code v expected)
  (define c (code v))
  (if (>= c 0)
      c
      (logic-error "~a: ~a: expected ~a, got ~e" who key expected v)))

(define (ratio-value who r)
  (if (and (real? r) (<= 0 r 1))
      (real->double-flonum r)
      (logic-error "~a: l1-ratio: expected a real in [0, 1], got ~e" who r)))

(define (seed-value who s)
  (if (or (not s) (and (exact-integer? s) (<= 0 s seed-most)))
      s
      (logic-error "~a: random-state: expected #f or an integer in [0, ~a], got ~e" who seed-most s)))

(struct sgd-settings
        (penalty alpha
                 l1-ratio
                 fit-intercept
                 epochs
                 tol
                 shuffle
                 learning-rate
                 eta0
                 power-t
                 batch-size
                 n-iter-no-change
                 random-state))

(define (sgd-settings/who who params)
  (define (ref key)
    (hash-ref params key))
  (sgd-settings (coded who 'penalty penalty-code (ref 'penalty) "one of 'none 'l1 'l2 'elasticnet")
                (positive-value/who who 'alpha (ref 'alpha))
                (ratio-value who (ref 'l1-ratio))
                (intercept-flag/who who (ref 'fit-intercept))
                (bounded-count/who who 'epochs (ref 'epochs))
                (nonnegative-value/who who 'tol (ref 'tol))
                (flag/who who 'shuffle (ref 'shuffle))
                (coded who
                       'learning-rate
                       learning-rate-code
                       (ref 'learning-rate)
                       "one of 'constant 'invscaling 'adaptive")
                (positive-value/who who 'eta0 (ref 'eta0))
                (nonnegative-value/who who 'power-t (ref 'power-t))
                (bounded-count/who who 'batch-size (ref 'batch-size))
                (bounded-count/who who 'n-iter-no-change (ref 'n-iter-no-change))
                (seed-value who (ref 'random-state))))

(define (fit-job who resources X y out settings #:classifier [classifier #f])
  (match-define (struct* sgd-settings
                         ([penalty penalty] [alpha alpha]
                                            [l1-ratio l1-ratio]
                                            [fit-intercept fit-intercept]
                                            [epochs epochs]
                                            [tol tol]
                                            [shuffle shuffle]
                                            [learning-rate learning-rate]
                                            [eta0 eta0]
                                            [power-t power-t]
                                            [batch-size batch-size]
                                            [n-iter-no-change n-iter-no-change]
                                            [random-state random-state]))
    settings)
  (define seeded (if random-state 1 0))
  (define seed (or random-state 0))
  (match classifier
    [#f
     (cuml-linear-model-mbsgd-regressor-fit-job who
                                                resources
                                                X
                                                y
                                                out
                                                penalty
                                                alpha
                                                l1-ratio
                                                fit-intercept
                                                epochs
                                                tol
                                                shuffle
                                                learning-rate
                                                eta0
                                                power-t
                                                batch-size
                                                n-iter-no-change
                                                seeded
                                                seed)]
    [(list host? loss)
     (cuml-linear-model-mbsgd-classifier-fit-job who
                                                 resources
                                                 X
                                                 y
                                                 host?
                                                 out
                                                 loss
                                                 penalty
                                                 alpha
                                                 l1-ratio
                                                 fit-intercept
                                                 epochs
                                                 tol
                                                 shuffle
                                                 learning-rate
                                                 eta0
                                                 power-t
                                                 batch-size
                                                 n-iter-no-change
                                                 seeded
                                                 seed)]))

(define (require-X who X)
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X)))

(define (scalar v)
  (match-define (list x) (device-vector->list v))
  (real->double-flonum x))

(define/who (mbsgd-regressor-fit who
                                 X
                                 y
                                 #:hyperparameters mbsgd-regressor-params
                                 #:resources [resources (current-device-resources)])
  (require-X who X)
  (unless (device-vector? y)
    (logic-error "~a: y: expected a device-vector, got ~e" who y))
  (unless (eq? (dtype y) (dtype X))
    (logic-error "~a: y: expected dtype ~a (X's), got ~a" who (dtype X) (dtype y)))
  (unless (eq? loss 'squared-loss)
    (logic-error "~a: loss: expected 'squared-loss, got ~e" who loss))
  (define settings (sgd-settings/who who (bound-params mbsgd-regressor-params)))
  (define out (deposit-for/who who resources))
  (fit-job who resources X y out settings)
  (match-define (list coef intercept) (deposit->list out))
  (mbsgd-regressor-result coef (scalar intercept)))

(define (host-codes who kind y resources)
  (cond
    [(device-kind? kind) (values y #f)]
    [else
     (define-values (codes classes _weights) (fit-labels/who who kind y #f resources))
     (unless (= (length classes) 2)
       (logic-error "~a: y: expected exactly 2 classes, got ~a" who (length classes)))
     (values codes classes)]))

(define/who (mbsgd-classifier-fit who
                                  X
                                  y
                                  #:hyperparameters mbsgd-classifier-params
                                  #:resources [resources (current-device-resources)])
  (require-X who X)
  (define kind (y-kind/who who y))
  (define code (coded who 'loss loss-code loss "one of 'hinge 'log 'squared-loss"))
  (define settings (sgd-settings/who who (bound-params mbsgd-classifier-params)))
  (define-values (y-view classes) (host-codes who kind y resources))
  (define out (deposit-for/who who resources))
  (fit-job who resources X y-view out settings #:classifier (list (if classes 1 0) code))
  (match (deposit->list out)
    [(list coef intercept) (mbsgd-classifier-result coef (scalar intercept) classes kind loss)]
    [(list coef intercept device-classes)
     (mbsgd-classifier-result coef
                              (scalar intercept)
                              (device-vector->list device-classes)
                              kind
                              loss)]))

(define/who (mbsgd-regressor-predict who result X #:resources [resources (current-device-resources)])
  (define coef (checked-coef/who who result X mbsgd-regressor-result? mbsgd-regressor-result-coef))
  (glm-predict/who who coef (mbsgd-regressor-result-intercept result) X resources))

(define/who (mbsgd-classifier-decision-function who
                                                result
                                                X
                                                #:resources [resources (current-device-resources)])
  (define coef (checked-coef/who who result X mbsgd-classifier-result? mbsgd-classifier-result-coef))
  (glm-predict/who who coef (mbsgd-classifier-result-intercept result) X resources))

(define (threshold loss)
  (if (eq? loss 'squared-loss) 0.5 0.0))

(define/who (mbsgd-classifier-predict who result X #:resources [resources (current-device-resources)])
  (define z (mbsgd-classifier-decision-function/who who result X #:resources resources))
  (match-define (struct* mbsgd-classifier-result ([classes classes] [label-kind kind] [loss loss]))
    result)
  (define out (deposit-for/who who resources))
  (cuml-linear-model-threshold-labels-job who
                                          resources
                                          z
                                          (threshold loss)
                                          (class-view kind classes resources)
                                          out)
  (match-define (list labels) (deposit->list out))
  (if (list-kind? kind)
      (decode-labels classes labels)
      labels))
