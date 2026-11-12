#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre id))
         (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters bound-params define-codes define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "glm.rkt"
                  alpha-value/who
                  glm-predict/who
                  intercept-flag/who
                  iteration-count/who
                  real-value/who)
         (only-in "native.rkt" cuml-linear-model-elastic-net-fit-job))

(provide (struct-out elastic-net-result)
         (struct-out lasso-result)
         elastic-net-fit
         elastic-net-fit/who
         elastic-net-params
         elastic-net-predict
         elastic-net-predict/who
         lasso-fit
         lasso-fit/who
         lasso-params
         lasso-predict
         lasso-predict/who)

(define-syntax-parse-rule (define-cd-result result:id predict:id)
  #:with result? (format-id #'result "~a?" #'result)
  (begin
    (struct result (coef intercept n-iter))
    (define/who (predict who fitted X #:resources [resources (current-device-resources)])
      (match fitted
        [(result coef intercept _) (glm-predict/who who coef intercept X resources)]
        [_ (raise-argument-error who (symbol->string 'result?) fitted)]))))

(define-cd-result elastic-net-result elastic-net-predict)
(define-cd-result lasso-result lasso-predict)

(define-hyperparameters elastic-net-params
  [alpha 1.0]
  [l1-ratio 0.5]
  [fit-intercept #t]
  [max-iter 1000]
  [tol 1e-3]
  [selection 'cyclic])

(define-hyperparameters lasso-params
  [alpha 1.0]
  [fit-intercept #t]
  [max-iter 1000]
  [tol 1e-3]
  [selection 'cyclic])

(define-codes selection-code [cyclic 0] [random 1])

(define (cd-fit/who who result X y sample-weight resources hyperparameters)
  (define (hyperparameter key)
    (hash-ref hyperparameters key))
  (define flag (intercept-flag/who who (hyperparameter 'fit-intercept)))
  (define penalty (alpha-value/who who (hyperparameter 'alpha)))
  (define ratio (real-value/who who 'l1-ratio (hyperparameter 'l1-ratio)))
  (define epochs (iteration-count/who who (hyperparameter 'max-iter)))
  (define tolerance (real-value/who who 'tol (hyperparameter 'tol)))
  (define out (deposit-for/who who resources))
  (define-values (intercept n-iter)
    (cuml-linear-model-elastic-net-fit-job who
                                           resources
                                           X
                                           y
                                           sample-weight
                                           out
                                           penalty
                                           ratio
                                           flag
                                           epochs
                                           tolerance
                                           (selection-code (hyperparameter 'selection))))
  (match-define (list coef) (deposit->list out))
  (result coef intercept n-iter))

(define/who (elastic-net-fit who
                             X
                             y
                             #:sample-weight [sample-weight #f]
                             #:hyperparameters elastic-net-params
                             #:resources [resources (current-device-resources)])
  (cd-fit/who who elastic-net-result X y sample-weight resources (bound-params elastic-net-params)))

(define/who (lasso-fit who
                       X
                       y
                       #:sample-weight [sample-weight #f]
                       #:hyperparameters lasso-params
                       #:resources [resources (current-device-resources)])
  (cd-fit/who who
              lasso-result
              X
              y
              sample-weight
              resources
              (hash-set (bound-params lasso-params) 'l1-ratio 1.0)))
