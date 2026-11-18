#lang racket/base

(require (only-in cuml/linear-model/private/logistic-regression
                  logistic-regression-decision-function/who
                  logistic-regression-fit/who
                  logistic-regression-params
                  logistic-regression-predict-log-proba/who
                  logistic-regression-predict-proba/who
                  logistic-regression-predict/who)
         (only-in "estimator-support.rkt" accuracy-scorer define-predictor))

(provide logistic-regression
         logistic-regression-model
         logistic-regression?)

(define-predictor logistic-regression
  #:hyperparameters logistic-regression-params
  #:fit logistic-regression-fit/who
  #:predict logistic-regression-predict/who
  #:scorer accuracy-scorer
  #:predict-proba logistic-regression-predict-proba/who
  #:predict-log-proba logistic-regression-predict-log-proba/who
  #:decision-function logistic-regression-decision-function/who)
