#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in raft current-device-resources))

(provide (struct-out standard-scaler-result)
         current-scratch-elements
         standard-scaler-fit
         standard-scaler-fit-transform
         standard-scaler-fit-transform/who
         standard-scaler-fit/who
         standard-scaler-inverse-transform
         standard-scaler-inverse-transform/who
         standard-scaler-params
         standard-scaler-transform
         standard-scaler-transform/who)

(struct standard-scaler-result (mean var scale n-samples n-features with-mean with-std))

(define-hyperparameters standard-scaler-params
  [with-mean #t]
  [with-std #t])

(define (scratch-elements v)
  (cond
    [(and (exact-integer? v) (<= 1 v (expt 2 62))) v]
    [else (raise-argument-error 'current-scratch-elements "(integer-in 1 (expt 2 62))" v)]))

(define current-scratch-elements (make-parameter (expt 2 26) scratch-elements 'current-scratch-elements))

(define/who (standard-scaler-fit who
                                 X
                                 #:hyperparameters standard-scaler-params
                                 #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (standard-scaler-transform who
                                       result
                                       X
                                       #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (standard-scaler-inverse-transform who
                                               result
                                               X
                                               #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (standard-scaler-fit-transform who
                                           X
                                           #:hyperparameters standard-scaler-params
                                           #:resources [resources (current-device-resources)])
  (error 'unimplemented))
