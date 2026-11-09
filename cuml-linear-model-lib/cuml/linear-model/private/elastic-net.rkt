#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
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

(struct elastic-net-result (coef intercept n-iter))
(struct lasso-result (coef intercept n-iter))

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

(define (cd-fit/who who
                    result
                    X
                    y
                    sample-weight
                    resources
                    alpha
                    l1-ratio
                    fit-intercept
                    max-iter
                    tol
                    selection)
  (define flag (intercept-flag/who who fit-intercept))
  (define penalty (alpha-value/who who alpha))
  (define ratio (real-value/who who 'l1-ratio l1-ratio))
  (define epochs (iteration-count/who who max-iter))
  (define tolerance (real-value/who who 'tol tol))
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
                                           (selection-code selection)))
  (match-define (list coef) (deposit->list out))
  (result coef intercept n-iter))

(define/who (elastic-net-fit who
                             X
                             y
                             #:sample-weight [sample-weight #f]
                             #:hyperparameters elastic-net-params
                             #:resources [resources (current-device-resources)])
  (cd-fit/who who
              elastic-net-result
              X
              y
              sample-weight
              resources
              alpha
              l1-ratio
              fit-intercept
              max-iter
              tol
              selection))

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
              alpha
              1.0
              fit-intercept
              max-iter
              tol
              selection))

(define/who (elastic-net-predict who result X #:resources [resources (current-device-resources)])
  (match result
    [(elastic-net-result coef intercept _) (glm-predict/who who coef intercept X resources)]
    [_ (raise-argument-error who "elastic-net-result?" result)]))

(define/who (lasso-predict who result X #:resources [resources (current-device-resources)])
  (match result
    [(lasso-result coef intercept _) (glm-predict/who who coef intercept X resources)]
    [_ (raise-argument-error who "lasso-result?" result)]))
