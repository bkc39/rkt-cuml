#lang racket/base

(require (only-in racket/generic define-generics define/generic raise-support-error)
         (only-in racket/match match*))

(provide gen:estimator
         estimator?
         fit
         fitted?
         estimator-params
         (rename-out [set-params estimator-set-params])
         gen:predictor
         predictor?
         predict
         gen:clusterer
         clusterer?
         fit-predict
         gen:transformer
         transformer?
         transform
         fit-transform
         gen:scorer
         scorer?
         score)

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

(define (keyword->symbol kw)
  (string->symbol (keyword->string kw)))

(define set-params
  (procedure-rename
   (make-keyword-procedure
    (lambda (kws vals estimator [changes (hasheq)])
      (match* (kws changes)
        [('() _) (estimator-set-params estimator changes)]
        [(_ (? hash?)) (error 'estimator-set-params "unimplemented")]
        [(_ _) (raise-argument-error 'estimator-set-params "hash?" changes)])))
   'estimator-set-params))
