#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match match-define struct*)
         (only-in raft current-device-resources device-matrix? device-vector->list device-vector? dtype shape)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt" bounded-count/who flag/who glm-predict/who intercept-flag/who)
         (only-in "labels.rkt" class-view device-kind? fit-labels label-codes label-kind? list-kind? y-kind)
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

(define losses '(squared-loss log hinge))

(struct mbsgd-regressor-result (coef intercept) #:transparent)

(struct mbsgd-classifier-result (coef intercept classes label-kind loss)
  #:guard
  (lambda (coef intercept classes kind loss name)
    (cond
      [(not (device-vector? coef)) (logic-error "~a: coef: expected a device-vector, got ~e" name coef)]
      [(not (flonum? intercept))
       (logic-error "~a: intercept: expected a flonum, got ~e" name intercept)]
      [(not (and (list? classes) (= (length classes) 2)))
       (logic-error "~a: classes: expected a list of 2 labels, got ~e" name classes)]
      [(not (label-kind? kind))
       (logic-error "~a: label-kind: expected the label kind of a fitted result, got ~e" name kind)]
      [(not (memq loss losses))
       (logic-error "~a: loss: expected one of 'hinge 'log 'squared-loss, got ~e" name loss)]
      [else (values coef intercept classes kind loss)]))
  #:property prop:label-codes
  (lambda (who result y resources)
    (match-define (struct* mbsgd-classifier-result ([classes classes] [label-kind kind])) result)
    (label-codes who classes kind y resources)))

(define-hyperparameters mbsgd-regressor-params
  [loss 'squared-loss]
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
  [random-state #f])

(define-hyperparameters mbsgd-classifier-params
  [loss 'hinge]
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
  [random-state #f])

(define-codes loss-code [squared-loss 0] [log 1] [hinge 2])
(define-codes penalty-code [none 0] [l1 1] [l2 2] [elasticnet 3])
(define-codes learning-rate-code [constant 1] [invscaling 2] [adaptive 3])

(define seed-most (sub1 (expt 2 32)))

(define (coded who key code v expected)
  (define c (code v))
  (if (>= c 0)
      c
      (logic-error "~a: ~a: expected ~a, got ~e" who key expected v)))

(define (flonum-of v)
  (and (real? v) (real->double-flonum v)))

(define (positive-value who key v)
  (define f (flonum-of v))
  (if (and f (< 0.0 f +inf.0))
      f
      (logic-error "~a: ~a: expected a positive finite real, got ~e" who key v)))

(define (nonnegative-value who key v)
  (define f (flonum-of v))
  (if (and f (<= 0.0 f) (< f +inf.0))
      f
      (logic-error "~a: ~a: expected a nonnegative finite real, got ~e" who key v)))

(define (ratio-value who r)
  (if (and (real? r) (<= 0 r 1))
      (real->double-flonum r)
      (logic-error "~a: l1-ratio: expected a real in [0, 1], got ~e" who r)))

(define (seed-values who s)
  (cond
    [(not s) (list 0 0)]
    [(and (exact-integer? s) (<= 0 s seed-most)) (list 1 s)]
    [else
     (logic-error "~a: random-state: expected #f or an integer in [0, ~a], got ~e" who seed-most s)]))

(define (sgd-settings/who who params)
  (define (ref key)
    (hash-ref params key))
  (define penalty
    (coded who 'penalty penalty-code (ref 'penalty) "one of 'none 'l1 'l2 'elasticnet"))
  (define alpha (positive-value who 'alpha (ref 'alpha)))
  (define l1-ratio (ratio-value who (ref 'l1-ratio)))
  (define fit-intercept (intercept-flag/who who (ref 'fit-intercept)))
  (define epochs (bounded-count/who who 'epochs (ref 'epochs)))
  (define tol (nonnegative-value who 'tol (ref 'tol)))
  (define shuffle (flag/who who 'shuffle (ref 'shuffle)))
  (define learning-rate
    (coded who
           'learning-rate
           learning-rate-code
           (ref 'learning-rate)
           "one of 'constant 'invscaling 'adaptive"))
  (define eta0 (positive-value who 'eta0 (ref 'eta0)))
  (define power-t (nonnegative-value who 'power-t (ref 'power-t)))
  (define batch-size (bounded-count/who who 'batch-size (ref 'batch-size)))
  (define n-iter-no-change (bounded-count/who who 'n-iter-no-change (ref 'n-iter-no-change)))
  (append (list penalty
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
                n-iter-no-change)
          (seed-values who (ref 'random-state))))

(define (require-X who X)
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X)))

(define (scalar v)
  (real->double-flonum (car (device-vector->list v))))

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
  (apply cuml-linear-model-mbsgd-regressor-fit-job who resources X y out settings)
  (match-define (list coef intercept) (deposit->list out))
  (mbsgd-regressor-result coef (scalar intercept)))

(define (host-codes who kind y resources)
  (match kind
    [(? device-kind?) (values y #f)]
    [_
     (define-values (codes classes _weights) (fit-labels who kind y #f resources))
     (unless (= (length classes) 2)
       (logic-error "~a: y: expected exactly 2 classes, got ~a" who (length classes)))
     (values codes classes)]))

(define/who (mbsgd-classifier-fit who
                                  X
                                  y
                                  #:hyperparameters mbsgd-classifier-params
                                  #:resources [resources (current-device-resources)])
  (require-X who X)
  (define kind (y-kind who y))
  (define code (coded who 'loss loss-code loss "one of 'hinge 'log 'squared-loss"))
  (define settings (sgd-settings/who who (bound-params mbsgd-classifier-params)))
  (define-values (y-view classes) (host-codes who kind y resources))
  (define out (deposit-for/who who resources))
  (apply cuml-linear-model-mbsgd-classifier-fit-job
         who
         resources
         X
         y-view
         (if classes 1 0)
         out
         code
         settings)
  (match (deposit->list out)
    [(list coef intercept) (mbsgd-classifier-result coef (scalar intercept) classes kind loss)]
    [(list coef intercept device-classes)
     (mbsgd-classifier-result coef (scalar intercept) (device-vector->list device-classes) kind loss)]))

(define (checked-coef who result X result? name coef-of)
  (unless (result? result)
    (raise-argument-error who name result))
  (require-X who X)
  (define coef (coef-of result))
  (match-define (list p) (shape coef))
  (match-define (list _ d) (shape X))
  (unless (= p d)
    (logic-error "~a: X: expected ~a columns, got ~a" who p d))
  (unless (eq? (dtype coef) (dtype X))
    (logic-error "~a: X: expected dtype ~a (the fit's), got ~a" who (dtype coef) (dtype X)))
  coef)

(define/who (mbsgd-regressor-predict who result X #:resources [resources (current-device-resources)])
  (define coef
    (checked-coef who result X mbsgd-regressor-result? "mbsgd-regressor-result?" mbsgd-regressor-result-coef))
  (glm-predict/who who coef (mbsgd-regressor-result-intercept result) X resources))

(define (decision who result X resources)
  (define coef
    (checked-coef who
                  result
                  X
                  mbsgd-classifier-result?
                  "mbsgd-classifier-result?"
                  mbsgd-classifier-result-coef))
  (glm-predict/who who coef (mbsgd-classifier-result-intercept result) X resources))

(define/who (mbsgd-classifier-decision-function who
                                                result
                                                X
                                                #:resources [resources (current-device-resources)])
  (decision who result X resources))

(define (threshold loss)
  (if (eq? loss 'squared-loss) 0.5 0.0))

(define/who (mbsgd-classifier-predict who
                                      result
                                      X
                                      #:resources [resources (current-device-resources)])
  (define z (decision who result X resources))
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
  (cond
    [(list-kind? kind)
     (define table (list->vector classes))
     (map (lambda (c) (vector-ref table c)) (device-vector->list labels))]
    [else labels]))
