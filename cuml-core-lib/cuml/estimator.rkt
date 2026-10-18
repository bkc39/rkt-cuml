#lang racket/base

(require (only-in racket/generic define-generics define/generic raise-support-error))

(provide gen:estimator
         estimator?
         fit ;; noqa
         fitted? ;; noqa
         estimator-params ;; noqa
         estimator-set-params ;; noqa
         gen:predictor
         predictor?
         predict
         gen:clusterer
         clusterer?
         fit-predict ;; noqa
         gen:transformer
         transformer?
         transform
         fit-transform ;; noqa
         gen:scorer
         scorer?
         score) ;; noqa

(define-generics estimator
                 (fit estimator X [y] #:sample-weight [sample-weight])
                 (fitted? estimator)
                 (estimator-params estimator)
                 (estimator-set-params estimator params)
                 #:requires (fit fitted? estimator-params estimator-set-params))

(define-generics predictor (predict predictor X) #:requires (predict))

(define-generics
 clusterer
 (fit-predict clusterer X [y] #:sample-weight [sample-weight])
 #:fallbacks
 [(define (fit-predict clusterer X [y #f] #:sample-weight [sample-weight #f])
    (cond
      [(predictor? clusterer) (predict (fit clusterer X y #:sample-weight sample-weight) X)]
      [else (raise-support-error 'fit-predict clusterer)]))])

(define-generics transformer
                 (transform transformer X)
                 (fit-transform transformer X [y] #:sample-weight [sample-weight])
                 #:requires (transform)
                 #:fallbacks
                 [(define/generic generic-transform transform)
                  (define (fit-transform transformer X [y #f] #:sample-weight [sample-weight #f])
                    (generic-transform (fit transformer X y #:sample-weight sample-weight) X))])

(define-generics scorer (score scorer X [y] #:sample-weight [sample-weight]) #:requires (score))
