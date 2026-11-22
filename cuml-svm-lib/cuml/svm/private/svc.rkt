#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in cuml/core/private/label-codes prop:label-codes)
         (only-in cuml/core/private/labels label-codes/who label-kind?)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match-define struct*)
         (only-in raft current-device-resources))

(provide (struct-out svc-result)
         svc-decision-function
         svc-decision-function/who
         svc-fit
         svc-fit/who
         svc-params
         svc-predict
         svc-predict/who)

(define svm-label-jobs #f)

(struct svc-result
        (support-vectors support
                         dual-coef
                         intercept
                         n-support
                         classes
                         label-kind
                         n-iter
                         kernel
                         degree
                         gamma
                         coef0
                         cache-size)
  #:guard
  (lambda (sv s dual b n-support classes kind n-iter kernel degree gamma coef0 cache-size name)
    (if (label-kind? kind)
        (values sv s dual b n-support classes kind n-iter kernel degree gamma coef0 cache-size)
        (logic-error "~a: label-kind: expected the label kind of a fitted result, got ~e" name kind)))
  #:property prop:label-codes
  (lambda (who result y resources)
    (match-define (struct* svc-result ([classes classes] [label-kind kind])) result)
    (label-codes/who who svm-label-jobs classes kind y resources)))

(define-hyperparameters svc-params
  [kernel 'rbf]
  [degree 3]
  [gamma 'scale]
  [coef0 0.0]
  [C 1.0]
  [tol 1e-3]
  [cache-size 1024.0]
  [max-iter #f]
  [nochange-steps 1000])

(define/who (svc-fit who
                     X
                     y
                     #:sample-weight [sample-weight #f]
                     #:hyperparameters svc-params
                     #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (svc-decision-function who result X #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (svc-predict who result X #:resources [resources (current-device-resources)])
  (error 'unimplemented))
