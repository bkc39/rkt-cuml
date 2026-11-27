#lang racket/base

(require (only-in cuml/svm/private/svc
                  svc-decision-function/who
                  svc-fit/who
                  svc-params
                  svc-predict/who)
         (only-in "estimator-support.rkt" accuracy-scorer define-predictor))

(provide svc
         svc-model
         svc?)

(define-predictor svc
  #:hyperparameters svc-params
  #:fit svc-fit/who
  #:predict svc-predict/who
  #:scorer accuracy-scorer
  #:decision-function svc-decision-function/who)
