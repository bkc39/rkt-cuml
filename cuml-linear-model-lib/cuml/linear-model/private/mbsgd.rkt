#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in raft current-device-resources))

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

(struct mbsgd-classifier-result (coef intercept classes label-kind loss))

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

(define/who (mbsgd-regressor-fit who
                                 X
                                 y
                                 #:hyperparameters mbsgd-regressor-params
                                 #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (mbsgd-regressor-predict who result X #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (mbsgd-classifier-fit who
                                  X
                                  y
                                  #:hyperparameters mbsgd-classifier-params
                                  #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (mbsgd-classifier-decision-function who
                                                result
                                                X
                                                #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (mbsgd-classifier-predict who
                                      result
                                      X
                                      #:resources [resources (current-device-resources)])
  (error 'unimplemented))
