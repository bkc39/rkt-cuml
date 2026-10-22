#lang racket/base

(require (only-in cuml/cluster/private/define-who define/who)
         (only-in cuml/cluster/private/hyperparameters define-hyperparameters)
         (only-in racket/list combinations)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn check-pred check-regexp-match check-true test-case)
         (only-in "private/cuml-error.rkt" check-cuml-error raised))

(define (twice size)
  (* 2 size))

(define-hyperparameters widget-params
  [size 3]
  [scale (twice size)]
  [shade 'plain #:fixed]
  [label "w"])

(define/who (spy who X #:hyperparameters widget-params #:extra [extra 0])
  (list who X (hasheq 'size size 'scale scale 'shade shade 'label label) extra))

(define/who (skipping who X #:hyperparameters widget-params #:ignoring (shade label))
  (list who X size scale))

(define literal (hasheq 'size 3 'scale 6 'shade 'plain 'label "w"))

(define changed (hasheq 'size 5 'scale 7 'shade 'bold 'label "x"))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(define (call-with proc keys table positional)
  (define sorted (sort keys keyword<? #:key keyword-of))
  (keyword-apply proc
                 (map keyword-of sorted)
                 (map (lambda (k) (hash-ref table k)) sorted)
                 positional))

(define (subsets keys n)
  (for*/list ([k (in-range (add1 n))]
              [subset (in-list (combinations keys k))])
    subset))

(define entry-keys '(size scale label))

(test-case "H1: the defaults are the literal table, an immutable hasheq with the table's keys"
  (check-equal? (widget-params) literal)
  (check-true (immutable? (widget-params)))
  (check-true (hash-eq? (widget-params)))
  (check-equal? (sort (hash-keys (widget-params)) symbol<?) '(label scale shade size))
  (check-equal? (object-name widget-params) 'widget-params))

(test-case "H2: one keyword replaces its key; the dependent default follows its expression"
  (for ([key (in-list '(size scale shade label))])
    (define value (hash-ref changed key))
    (define expected
      (match key
        ['size (hash-set* literal 'size value 'scale (twice value))]
        [_ (hash-set literal key value)]))
    (check-equal? (call-with widget-params (list key) changed '()) expected (format "~a" key))))

(test-case "H3: an override of an override is hash-set*, and a base's dependent default stays"
  (for* ([first (in-list (subsets '(size scale shade label) 2))]
         [second (in-list (subsets '(size scale shade label) 2))])
    (define base (call-with widget-params first changed '()))
    (define over
      (call-with widget-params second (hasheq 'size 11 'scale 13 'shade 'x 'label "y") (list base)))
    (check-equal?
     over
     (for/fold ([set base]) ([key (in-list second)])
       (hash-set set key (hash-ref (hasheq 'size 11 'scale 13 'shade 'x 'label "y") key)))))
  (check-equal? (hash-ref (widget-params (widget-params) #:size 10) 'scale) 6))

(test-case "H4: a set resolves to itself, and #f to the defaults"
  (for ([keys (in-list (subsets '(size scale shade label) 2))])
    (define s (call-with widget-params keys changed '()))
    (check-equal? (widget-params s) s))
  (check-equal? (widget-params #f) (widget-params)))

(test-case "H5: the messages of a bad base"
  (check-cuml-error 'logic
                    "widget-params: widget has no parameter colour"
                    (lambda () (widget-params (hash-set literal 'colour 'red))))
  (check-cuml-error 'logic
                    "widget-params: expected every widget parameter, missing size"
                    (lambda () (widget-params (hash-remove literal 'size))))
  (check-cuml-error 'logic
                    "widget-params: widget has no parameter \"size\""
                    (lambda () (widget-params (hash-set literal "size" 1))))
  (define e (raised (lambda () (widget-params 7))))
  (check-pred exn:fail:contract? e)
  (check-regexp-match #rx"^widget-params: contract violation\n  expected: \\(or/c #f hash\\?\\)"
                      (exn-message e)))

(test-case "H6: a keyword is the same as a set with that keyword, with or without a base"
  (for* ([keys (in-list (subsets entry-keys 2))]
         [base (in-list (list #f (widget-params #:size 9 #:label "b")))])
    (check-equal? (call-with spy keys changed (list 'X))
                  (spy 'X #:params (call-with widget-params keys changed '())))
    (check-equal? (call-with spy (cons 'params keys) (hash-set changed 'params base) (list 'X))
                  (spy 'X #:params (call-with widget-params keys changed (list base))))))

(define (keywords-of proc)
  (call-with-values (lambda () (procedure-keywords proc)) list))

(test-case "H7: the entry point names itself, checks the fixed key and takes the table's keywords"
  (check-cuml-error 'logic
                    "spy: widget has no parameter colour"
                    (lambda () (spy 'X #:params (hash-set literal 'colour 'red))))
  (check-cuml-error 'logic
                    "spy: expected every widget parameter, missing size"
                    (lambda () (spy 'X #:params (hash-remove literal 'size))))
  (check-pred exn:fail:contract? (raised (lambda () (spy 'X #:params 7))))
  (check-regexp-match #rx"^spy: " (exn-message (raised (lambda () (spy 'X #:params 7)))))
  (check-cuml-error 'logic
                    "spy: shade: expected plain, got bold"
                    (lambda () (spy 'X #:params (widget-params #:shade 'bold))))
  (check-equal? (skipping 'X #:params (widget-params #:shade 'bold #:label "z")) '(skipping X 3 6))
  (check-equal? (keywords-of spy) '(() (#:extra #:label #:params #:scale #:size)))
  (check-equal? (keywords-of spy/who) (keywords-of spy))
  (check-equal? (keywords-of skipping) '(() (#:params #:scale #:size))))

(define-namespace-anchor anchor)

(define (expanded form)
  (parameterize ([current-namespace (namespace-anchor->namespace anchor)])
    (expand form)))

(define (check-syntax-error form pattern)
  (define e (raised (lambda () (expanded form))))
  (check-pred exn:fail:syntax? e)
  (check-regexp-match pattern (exn-message e)))

(test-case "H8: the definitions refuse malformed tables and entry points"
  (check-syntax-error '(define-hyperparameters widget-table [size 1]) #rx"ending in -params")
  (check-syntax-error '(define-hyperparameters pair-params
                         [size 1]
                         [size 2])
                      #rx"duplicate")
  (check-syntax-error '(define-hyperparameters later-params
                         [scale (twice size)]
                         [size 1])
                      #rx"later key")
  (check-syntax-error '(define-hyperparameters fixed-params
                         [size 1]
                         [shade #:fixed])
                      #rx"#:fixed key needs a default")
  (check-syntax-error '(define/who (f who X #:hyperparameters widget-params #:ignoring (colour)) X)
                      #rx"widget-params has no key colour")
  (check-syntax-error '(define/who (f who X #:hyperparameters twice) X) #rx"define-hyperparameters"))

(test-case "H8: a misspelt keyword is a syntax error, and a run-time call Racket's error"
  (check-syntax-error '(widget-params #:colour 1) #rx"#:colour")
  (check-syntax-error '(spy 'X #:colour 1) #rx"#:colour")
  (define e (raised (lambda () (keyword-apply widget-params '(#:colour) '(1) '()))))
  (check-pred exn:fail:contract? e)
  (check-regexp-match
   #rx"^application: procedure does not expect an argument with given keyword\n  procedure: widget-params\n  given keyword: #:colour"
   (exn-message e))
  (check-exn exn:fail:contract? (lambda () ((values spy) 'X #:colour 1))))
