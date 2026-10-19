#lang racket/base

(require (only-in cuml/cluster/private/defaults define-defaults)
         (only-in cuml/private/estimator-support define-constructor)
         (only-in rackunit check-equal? check-exn test-case))

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
