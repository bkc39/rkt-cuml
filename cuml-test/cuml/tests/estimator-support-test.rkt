#lang racket/base

(require (only-in cuml/cluster/private/define-who define/who)
         (only-in cuml/cluster/private/hyperparameters define-hyperparameters)
         (only-in cuml/estimator
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fit-transform
                  fitted?
                  predict
                  score
                  transform)
         (only-in cuml/private/estimator-support define-estimator)
         (only-in racket/list combinations)
         (only-in racket/string string-append*)
         (only-in rackunit check-equal? check-false check-match check-not-equal? check-pred check-regexp-match check-true test-case)
         (only-in "private/cuml-error.rkt" check-cuml-error raised))

(define (twice size)
  (* 2 size))

(define-hyperparameters gadget-params
  [size 3]
  [scale (twice size)]
  [shade 'plain #:fixed]
  [label "w"])

(define (model who X params settings weight)
  (list 'model who X params settings weight))

(define/who (gadget-fit who
                        X
                        #:hyperparameters gadget-params
                        #:sample-weight [weight #f]
                        #:speed [speed #f]
                        #:place [place #f])
  (model who X (hasheq 'size size 'scale scale 'shade shade 'label label) (list speed place) weight))

(define applied-calls (box 0))

(define ((applied-spy kind) who argument X #:speed [speed #f] #:place [place #f] #:sample-weight [w 'none])
  (set-box! applied-calls (add1 (unbox applied-calls)))
  (list kind who argument X speed place w))

(define (labelled m)
  (list 'labels m))

(define-estimator gadget
  #:hyperparameters gadget-params
  #:fit gadget-fit/who
  #:labels labelled
  #:settings (speed place)
  #:no-sample-weight
  #:apply values
  [predict (applied-spy 'predict)]
  [transform (applied-spy 'transform)]
  [score (applied-spy 'score) #:sample-weight])

(define-estimator weighted-gadget
  #:hyperparameters gadget-params
  #:fit gadget-fit/who
  #:labels labelled
  #:settings (place)
  #:apply values
  [transform (applied-spy 'transform)]
  [score (applied-spy 'score)])

(define changed (hasheq 'size 5 'scale 7 'shade 'bold 'label "x" 'speed 2 'place 'here))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(define (call-with proc keys positional)
  (define sorted (sort keys keyword<? #:key keyword-of))
  (keyword-apply proc (map keyword-of sorted) (map (lambda (k) (hash-ref changed k)) sorted) positional))

(define (subsets keys n)
  (for*/list ([k (in-range (add1 n))]
              [subset (in-list (combinations keys k))])
    subset))

(define table-keys '(size scale shade label))

(test-case "E1: the constructor's parameters are the table's"
  (for ([keys (in-list (subsets table-keys 2))])
    (check-equal? (estimator-params (call-with gadget keys '())) (call-with gadget-params keys '()))
    (define base (gadget-params #:size 9))
    (check-equal? (estimator-params (call-with gadget keys (list base)))
                  (call-with gadget-params keys (list base)))))

(test-case "E2: estimator-set-params by keyword, by hash and through the table agree"
  (define e (fit (gadget #:label "a" #:speed 4) 'X))
  (for ([key (in-list table-keys)])
    (define value (hash-ref changed key))
    (define by-keyword (keyword-apply estimator-set-params (list (keyword-of key)) (list value) (list e)))
    (define by-hash (estimator-set-params e (hasheq key value)))
    (check-equal? (estimator-params by-keyword) (estimator-params by-hash))
    (check-equal? (estimator-params by-keyword)
                  (keyword-apply gadget-params (list (keyword-of key)) (list value) (list (estimator-params e))))
    (check-false (fitted? by-keyword))
    (check-equal? by-keyword (gadget (estimator-params by-keyword) #:speed 4)))
  (check-equal? (hash-ref (estimator-params (estimator-set-params e (hasheq 'size 1 'label "h") #:size 2)) 'size)
                2)
  (check-equal? (hash-ref (estimator-params (estimator-set-params e (hasheq 'size 1 'label "h") #:size 2)) 'label)
                "h"))

(define (expected-text keys)
  (define set (call-with gadget-params (filter (lambda (k) (memq k table-keys)) keys) '()))
  (define defaults (hasheq 'size 3 'scale (twice (hash-ref set 'size)) 'shade 'plain 'label "w"))
  (string-append "#<gadget"
                 (string-append* (for/list ([key (in-list table-keys)]
                                            #:unless (equal? (hash-ref set key) (hash-ref defaults key)))
                                   (format " #:~a ~v" key (hash-ref set key))))
                 (string-append* (for/list ([key (in-list '(speed place))]
                                            #:when (memq key keys))
                                   (format " #:~a ~v" key (hash-ref changed key))))
                 ">"))

(test-case "E3: the printer shows the keys that differ from their defaults, then the settings given"
  (for ([keys (in-list (subsets (append table-keys '(speed place)) 2))])
    (define e (call-with gadget keys '()))
    (check-equal? (format "~a" e) (expected-text keys) (format "~a" keys))
    (check-equal? (format "~v" e) (expected-text keys)))
  (check-equal? (format "~a" (gadget #:size 3 #:scale 6)) "#<gadget>")
  (check-equal? (format "~a" (gadget #:size 5)) "#<gadget #:size 5>")
  (check-equal? (format "~a" (fit (gadget #:speed 1) 'X)) "#<gadget fitted #:speed 1>"))

(test-case "E4: equality and hashing cover the parameters and settings, and the fit by eq?"
  (check-equal? (gadget #:size 4 #:speed 1) (gadget #:size 4 #:speed 1))
  (check-equal? (equal-hash-code (gadget #:size 4 #:speed 1)) (equal-hash-code (gadget #:size 4 #:speed 1)))
  (check-equal? (equal-secondary-hash-code (gadget #:size 4)) (equal-secondary-hash-code (gadget #:size 4)))
  (check-not-equal? (gadget #:size 4) (gadget #:size 5))
  (check-not-equal? (gadget #:speed 1) (gadget #:speed 2))
  (define f (fit (gadget) 'X))
  (check-equal? f f)
  (check-not-equal? f (fit (gadget) 'X))
  (check-not-equal? f (gadget)))

(test-case "E5: an estimator without sample weights refuses one"
  (check-cuml-error 'logic "fit: gadget takes no sample-weight" (lambda () (fit (gadget) 'X #:sample-weight 'w)))
  (check-cuml-error 'logic
                    "fit-predict: gadget takes no sample-weight"
                    (lambda () (fit-predict (gadget) 'X #:sample-weight 'w)))
  (check-match (cadr (fit-predict (weighted-gadget) 'X #:sample-weight 'w)) (list 'model _ _ _ _ 'w)))

(test-case "E5: fit passes the generic's name, the set and the settings given"
  (define m (cadr (fit-predict (gadget #:size 4 #:place 'p) 'X)))
  (check-equal? m (model 'fit-predict 'X (gadget-params #:size 4) (list #f 'p) #f))
  (check-cuml-error 'logic "fit: shade: expected plain, got bold" (lambda () (fit (gadget #:shade 'bold) 'X))))

(test-case "E6: the applied methods pass the generic's name and the settings"
  (define f (fit (gadget #:speed 2) 'X))
  (define m (model 'fit 'X (gadget-params) (list 2 #f) #f))
  (check-equal? (predict f 'Y) (list 'predict 'predict m 'Y 2 #f 'none))
  (check-equal? (transform f 'Y) (list 'transform 'transform m 'Y 2 #f 'none))
  (check-equal? (score f 'Y #:sample-weight 'w) (list 'score 'score m 'Y 2 #f 'w))
  (define g (weighted-gadget #:place 'p))
  (check-equal? (fit-transform g 'Y #:sample-weight 'w)
                (list 'transform 'fit-transform (model 'fit-transform 'Y (gadget-params) (list #f 'p) 'w) 'Y #f 'p 'none))
  (check-equal? (list-ref (score (fit g 'Y) 'Y #:sample-weight 'w) 6) 'none))

(test-case "E6: an unfitted estimator is refused before X is read"
  (set-box! applied-calls 0)
  (for ([generic (in-list (list predict transform score))]
        [who (in-list '(predict transform score))])
    (check-cuml-error 'logic (format "~a: gadget is not fitted" who) (lambda () (generic (gadget) 'X))))
  (check-equal? (unbox applied-calls) 0))

(define-namespace-anchor anchor)

(test-case "Q-C: a misspelt constructor keyword is a syntax error"
  (define e
    (raised (lambda ()
              (parameterize ([current-namespace (namespace-anchor->namespace anchor)])
                (expand '(gadget #:colour 1))))))
  (check-pred exn:fail:syntax? e)
  (check-regexp-match #rx"#:colour" (exn-message e))
  (check-true (procedure? gadget)))
