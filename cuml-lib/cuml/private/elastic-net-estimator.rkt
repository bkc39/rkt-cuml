#lang racket/base

(require (only-in cuml/linear-model/private/elastic-net
                  elastic-net-fit/who
                  elastic-net-params
                  elastic-net-predict/who
                  lasso-fit/who
                  lasso-params
                  lasso-predict/who)
         (only-in "estimator-support.rkt" define-regressor))

(provide elastic-net
         elastic-net-model
         elastic-net?
         lasso
         lasso-model
         lasso?)

(define-regressor lasso #:hyperparameters lasso-params #:fit lasso-fit/who #:predict lasso-predict/who)

(define-regressor elastic-net
  #:hyperparameters elastic-net-params
  #:fit elastic-net-fit/who
  #:predict elastic-net-predict/who)
