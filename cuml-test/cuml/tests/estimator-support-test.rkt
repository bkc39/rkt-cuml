#lang racket/base

(require (only-in cuml/cluster/private/defaults define-codes define-defaults)
         (only-in cuml/estimator
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fitted?
                  gen:predictor
                  predict
                  predictor?)
         (only-in cuml/private/estimator-support define-constructor define-estimator)
         (only-in rackunit check-equal? check-exn check-false check-not-equal? check-true test-case)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define-defaults widget-defaults [size 3] [label 'plain] [scale 1.5])

(struct widget (params model) #:transparent)

(define (doubled size)
  (* 2 size))

(define-constructor (widget/new #:size #:label #:scale (doubled size))
                    #:names widget-names
                    #:defaults widget-defaults
                    widget)

(test-case "define-defaults binds the table and default reads it"
  (check-equal? widget-defaults (hasheq 'size 3 'label 'plain 'scale 1.5))
  (check-equal? (default 'label) 'plain)
  (check-exn exn:fail:contract? (lambda () (default 'missing))))

(test-case "define-constructor names the parameters in order"
  (check-equal? widget-names '(size label scale)))

(test-case "an omitted parameter takes the table's value, and the model starts as #f"
  (check-equal? (widget/new #:scale 1.5) (widget (hasheq 'size 3 'label 'plain 'scale 1.5) #f))
  (check-equal? (widget/new #:label 'bold #:scale 0)
                (widget (hasheq 'size 3 'label 'bold 'scale 0) #f)))

(test-case "a parameter's own default expression replaces the table and sees earlier parameters"
  (check-equal? (hash-ref (widget-params (widget/new)) 'scale) 6)
  (check-equal? (hash-ref (widget-params (widget/new #:size 5)) 'scale) 10))

(define-codes shade-code [plain 0] [bold 1])

(test-case "define-codes maps each name to its code and anything else to -1"
  (check-equal? (map shade-code '(plain bold italic "plain" 0)) '(0 1 -1 -1 -1)))

(define (gadget-fitted who e X sample-weight)
  (list who (gadget-params e) X sample-weight))

(define gadget-names '(size label))

(define gadget-defaults (hasheq 'size 3 'label 3))

(define-estimator gadget
                  #:fitted gadget-fitted
                  #:labels cddr
                  #:names gadget-names
                  #:defaults gadget-defaults
                  #:shown-defaults
                  (lambda (params) (hash-set gadget-defaults 'size (hash-ref params 'label)))
                  #:methods gen:predictor
                  [(define (predict e X)
                     (list 'predicted (gadget-model e) X))])

(define plain (gadget* gadget-defaults #f))

(test-case "define-estimator: fit keeps the parameters and stores the fitted model"
  (define fitted (fit plain 'X #:sample-weight 'w))
  (check-equal? (gadget-model fitted) (list 'fit gadget-defaults 'X 'w))
  (check-equal? (gadget-params fitted) gadget-defaults)
  (check-true (fitted? fitted))
  (check-false (fitted? plain))
  (check-equal? (estimator-params fitted) gadget-defaults))

(test-case "define-estimator: fit-predict reads the labels of the fitted model"
  (check-equal? (fit-predict plain 'X) '(X #f)))

(test-case "define-estimator: set-params merges known names and drops the model"
  (define fitted (fit plain 'X))
  (check-equal? (estimator-set-params fitted (hasheq 'size 4)) (gadget* (hasheq 'size 4 'label 3) #f))
  (check-cuml-error 'logic
                    "estimator-set-params: gadget has no parameter colour"
                    (lambda () (estimator-set-params plain (hasheq 'colour 'red)))))

(test-case "define-estimator: equal on equal parameters and the same model"
  (define fitted (fit plain 'X))
  (check-equal? (gadget* (hasheq 'size 3 'label 3) #f) plain)
  (check-equal? (equal-hash-code (gadget* (hasheq 'size 3 'label 3) #f)) (equal-hash-code plain))
  (check-not-equal? fitted (fit plain 'X))
  (check-equal? fitted fitted))

(test-case "define-estimator: prints the fitted mark and the parameters off their shown defaults, in order"
  (check-equal? (format "~a" plain) "#<gadget>")
  (check-equal? (format "~a" (fit plain 'X)) "#<gadget fitted>")
  (check-equal? (format "~a" (gadget* (hasheq 'size 5 'label 5) #f)) "#<gadget #:label 5>")
  (check-equal? (format "~a" (gadget* (hasheq 'size 4 'label 'bold) #f))
                "#<gadget #:size 4 #:label 'bold>"))

(define-namespace-anchor anchor)

(test-case "define-estimator: a #:methods table that is not a list of definitions is refused at the macro"
  (check-exn #rx"define-estimator: expected a bracketed list of method definitions"
             (lambda ()
               (parameterize ([current-namespace (namespace-anchor->namespace anchor)])
                 (expand '(define-estimator doohickey
                                            #:fitted gadget-fitted
                                            #:labels cddr
                                            #:names gadget-names
                                            #:defaults gadget-defaults
                                            #:methods gen:predictor
                                            (predict e X)))))))

(test-case "define-estimator: the extra method tables are the estimator's"
  (check-true (predictor? plain))
  (check-equal? (predict (fit plain 'X) 'Y) (list 'predicted (list 'fit gadget-defaults 'X #f) 'Y)))
