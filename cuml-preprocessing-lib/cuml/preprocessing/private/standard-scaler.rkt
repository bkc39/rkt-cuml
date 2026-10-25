#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources shape)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-preprocessing-standard-scaler-fit-job
                  cuml-preprocessing-standard-scaler-inverse-transform-job
                  cuml-preprocessing-standard-scaler-transform-job))

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

(define (flag v)
  (if v 1 0))

(define (fitted who X with-mean with-std resources)
  (define out (deposit-for/who who resources))
  (define n-samples
    (cuml-preprocessing-standard-scaler-fit-job who
                                                resources
                                                X
                                                out
                                                (flag with-mean)
                                                (flag with-std)
                                                (current-scratch-elements)))
  (match-define (list _ n-features) (shape X))
  (define-values (mean var scale)
    (match (deposit->list out)
      [(list mean var scale) (values mean var scale)]
      [(list mean) (values (and with-mean mean) #f #f)]))
  (standard-scaler-result mean var scale n-samples n-features (and with-mean #t) (and with-std #t)))

(define (applied job who result X resources)
  (match-define (standard-scaler-result mean _ scale _ n-features with-mean with-std) result)
  (define out (deposit-for/who who resources))
  (job who resources X (and with-mean mean) (and with-std scale) out n-features)
  (match-define (list transformed) (deposit->list out))
  transformed)

(define/who (standard-scaler-fit who
                                 X
                                 #:hyperparameters standard-scaler-params
                                 #:resources [resources (current-device-resources)])
  (fitted who X with-mean with-std resources))

(define/who (standard-scaler-transform who
                                       result
                                       X
                                       #:resources [resources (current-device-resources)])
  (applied cuml-preprocessing-standard-scaler-transform-job who result X resources))

(define/who (standard-scaler-inverse-transform who
                                               result
                                               X
                                               #:resources [resources (current-device-resources)])
  (applied cuml-preprocessing-standard-scaler-inverse-transform-job who result X resources))

(define/who (standard-scaler-fit-transform who
                                           X
                                           #:hyperparameters standard-scaler-params
                                           #:resources [resources (current-device-resources)])
  (define result (fitted who X with-mean with-std resources))
  (values result
          (applied cuml-preprocessing-standard-scaler-transform-job who result X resources)))
