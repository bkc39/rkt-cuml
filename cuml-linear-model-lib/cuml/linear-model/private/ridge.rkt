#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt" alpha-value/who glm-predict/who intercept-flag/who)
         (only-in "native.rkt" cuml-linear-model-ridge-fit-job))

(provide (struct-out ridge-result)
         ridge-fit
         ridge-fit/who
         ridge-params
         ridge-predict
         ridge-predict/who)

(struct ridge-result (coef intercept))

(define-hyperparameters ridge-params
  [alpha 1.0]
  [solver 'auto]
  [fit-intercept #t])

(define-codes solver-code [auto 1] [eig 1] [svd 0])

(define/who (ridge-fit who
                       X
                       y
                       #:sample-weight [sample-weight #f]
                       #:hyperparameters ridge-params
                       #:resources [resources (current-device-resources)])
  (define flag (intercept-flag/who who fit-intercept))
  (define penalty (alpha-value/who who alpha))
  (define out (deposit-for/who who resources))
  (define intercept
    (cuml-linear-model-ridge-fit-job who
                                     resources
                                     X
                                     y
                                     sample-weight
                                     out
                                     penalty
                                     flag
                                     (solver-code solver)))
  (match-define (list coef) (deposit->list out))
  (ridge-result coef intercept))

(define/who (ridge-predict who result X #:resources [resources (current-device-resources)])
  (match result
    [(ridge-result coef intercept) (glm-predict/who who coef intercept X resources)]
    [_ (raise-argument-error who "ridge-result?" result)]))
