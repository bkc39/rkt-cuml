#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in raft current-device-resources))

(provide (struct-out tsvd-result)
         tsvd-fit
         tsvd-fit-transform
         tsvd-fit-transform/who
         tsvd-fit/who
         tsvd-inverse-transform
         tsvd-inverse-transform/who
         tsvd-params
         tsvd-transform
         tsvd-transform/who)

(struct tsvd-result (components explained-variance explained-variance-ratio singular-values))

(define-hyperparameters tsvd-params
  [n-components 1]
  [solver 'full]
  [tol 0.0]
  [iterations 15])

(define/who (tsvd-fit who
                      X
                      #:hyperparameters tsvd-params
                      #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (tsvd-transform who result X #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (tsvd-inverse-transform who result X #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (tsvd-fit-transform who
                                X
                                #:hyperparameters tsvd-params
                                #:resources [resources (current-device-resources)])
  (error 'unimplemented))
