#lang racket/base

(require (only-in racket/file make-temporary-file)
         (only-in rackunit check-equal? test-case))

(define (call-with-module-file forms proc)
  (define file (make-temporary-file "review-~a.rkt"))
  (dynamic-wind void
                (lambda ()
                  (with-output-to-file file
                                       #:exists 'truncate
                                       (lambda ()
                                         (displayln "#lang racket/base")
                                         (for-each writeln forms)))
                  (proc file))
                (lambda () (delete-file file))))

(define (problems-in forms)
  (call-with-module-file forms
                         (lambda (file)
                           (parameterize ([current-namespace (make-base-namespace)])
                             (map (dynamic-require 'review/problem 'problem-message)
                                  ((dynamic-require 'review/lint 'lint) file))))))

(define quoted-data '((provide form) (define form '(define-job deposit x))))

(define quasiquoted-data '((provide form) (define x 1) (define form `(define-job deposit ,x))))

(define unquotes-in-compound-data
  '((provide forms) (define x 1)
                    (define y 2)
                    (define z 3)
                    (define forms (list `#& ,x `#hash((a . ,y)) `#s(p ,z)))))

(define test-forms-with-same-names
  '((require (only-in raft/tests/private/gpu test-gpu)) (test-gpu "a"
                                                          (define y 1)
                                                          y)
                                                        (test-gpu "b"
                                                          (define y 2)
                                                          y)))

(define generics-with-optional-and-keyword-methods
  '((require racket/generic) (provide gen:shape
                                      shape?
                                      area
                                      scale)
                             (define-generics shape
                                              (area shape [units])
                                              (scale shape factor #:origin [origin])
                                              #:requires (area))))

(define constructor-renamed-struct
  '((provide point
             point?
             point-x
             point-y)
    (struct point (x y) #:constructor-name point* #:omit-define-syntaxes #:transparent)
    (define (point #:x [x 0] #:y [y 0])
      (point* x y))))

(define constructor-renamed-struct-with-syntaxes
  '((provide point
             point?
             point-x
             point-y)
    (struct point (x y) #:constructor-name point*)
    (define (point x y)
      (point* x y))))

(define who-definitions
  '((provide twice
             twice/who)
    (define-syntax-parse-rule (define/who . _)
      (void))
    (define/who (twice who x #:times [n 2]) (list who x n))))

(define who-definition-with-unused-local
  '((provide twice
             twice/who)
    (define-syntax-parse-rule (define/who . _)
      (void))
    (define/who (twice who x)
      (define unused 1)
      (list who x))))

(define splicing-syntax-class-used-in-a-pattern
  '((require (for-syntax racket/base
                         syntax/parse/pre))
    (provide m)
    (begin-for-syntax
      (define-splicing-syntax-class pair (pattern (~seq a b))))
    (define-syntax (m stx)
      (syntax-parse stx
        [(_ p:pair) #'(list p.a p.b)]))))

(test-case "a splicing syntax class is a syntax class: its uses are inside patterns"
  (check-equal? (problems-in splicing-syntax-class-used-in-a-pattern) '()))

(test-case "define-generics binds each method, whatever its arguments"
  (check-equal? (problems-in generics-with-optional-and-keyword-methods) '()))

(test-case "a struct whose syntaxes are omitted binds its constructor name, not its own"
  (check-equal? (problems-in constructor-renamed-struct) '()))

(test-case "a struct that keeps its syntaxes still binds its own name"
  (check-equal? (problems-in constructor-renamed-struct-with-syntaxes)
                '("identifier 'point' is already defined")))

(test-case "define/who binds the name and its /who form"
  (check-equal? (problems-in who-definitions) '()))

(test-case "a define/who body is reviewed as the /who form's"
  (check-equal? (problems-in who-definition-with-unused-local)
                '("identifier 'unused' is never used")))

(test-case "quoted data is not read as code"
  (check-equal? (problems-in quoted-data) '()))

(test-case "quasiquoted data is read only at its unquotes"
  (check-equal? (problems-in quasiquoted-data) '()))

(test-case "an unquote inside a quasiquoted box, hash value or prefab field is a use"
  (check-equal? (problems-in unquotes-in-compound-data) '()))

(test-case "each test form's body is a scope of its own"
  (check-equal? (problems-in test-forms-with-same-names) '()))

(define exported-who-procedure
  '((provide twice) (define (twice who x)
                      (list who x))))

(define renamed-exported-who-procedure
  '((provide (rename-out [twice double])) (define (twice who x)
                                            (list who x))))

(define private-relay
  '((provide twice) (define (helper who x)
                      (list who x))
                    (define (twice x)
                      (helper 'twice x))))

(define hyperparameter-table
  '((provide widget-params) (define (base-size)
                              3)
                            (define-hyperparameters widget-params
                              [size (base-size)]
                              [scale (* 2 size)]
                              [shade 'plain #:fixed])))

(define who-definition-with-a-table
  '((provide paint) (define-hyperparameters widget-params
                      [size 3]
                      [shade 'plain])
                    (define/who (paint who X #:hyperparameters widget-params #:ignoring (shade))
                      (list who X size))))

(define estimator-definition
  '((provide widget
             widget?
             widget-model)
    (define (widget-fit/who who X)
      (list who X))
    (define-estimator widget
      #:hyperparameters widget-params
      #:fit widget-fit/who
      #:labels values
      #:settings ())))

(define helper-used-in-a-template
  '((require (for-syntax racket/base)) (provide m)
                                       (define (helper x)
                                         x)
                                       (define-syntax-parse-rule (m x:expr)
                                         #:with y #'x
                                         (helper y))))

(test-case "an exported procedure that takes who first is defined with define/who"
  (check-equal? (problems-in exported-who-procedure)
                '("twice takes who first and is exported: define it with define/who"))
  (check-equal? (problems-in renamed-exported-who-procedure)
                '("twice takes who first and is exported: define it with define/who")))

(test-case "a private relay that takes who is not an entry point"
  (check-equal? (problems-in private-relay) '()))

(test-case "define-hyperparameters binds the table and reviews its defaults"
  (check-equal? (problems-in hyperparameter-table) '()))

(test-case "a define/who table is a use, and its keys are not formals"
  (check-equal? (problems-in who-definition-with-a-table) '()))

(test-case "define-estimator binds the constructor, the predicate and the model accessor"
  (check-equal? (problems-in estimator-definition) '()))

(test-case "a binding used only in a macro's template is used"
  (check-equal? (problems-in helper-used-in-a-template) '()))

(define hand-written-who-pair
  '((provide twice) (define (twice/who who x)
                      (list who x))
                    (define (twice x)
                      (twice/who 'twice x))))

(test-case "a procedure that only relays to its /who form is a define/who"
  (check-equal? (problems-in hand-written-who-pair)
                '("twice only relays to twice/who: define both with define/who")))

(define wrapper-that-prepares
  '((provide f) (define (f/who who x)
                  (list who x))
                (define (prep x)
                  x)
                (define (f x)
                  (f/who 'f (prep x)))))

(define who-procedure-under-all-defined-out
  '((provide (all-defined-out)) (define (twice who x)
                                  (list who x))))

(define who-procedure-under-prefix-out
  '((provide (prefix-out my: twice)
             thrice)
    (define (twice who x)
      (list who x))
    (define (thrice x)
      (twice 'thrice x))))

(test-case "L7: a wrapper that passes other arguments than its formals is not a relay"
  (check-equal? (problems-in wrapper-that-prepares) '()))

(test-case "L7: all-defined-out and prefix-out export a procedure that takes who"
  (check-equal? (problems-in who-procedure-under-all-defined-out)
                '("twice takes who first and is exported: define it with define/who"))
  (check-equal? (problems-in who-procedure-under-prefix-out)
                '("twice takes who first and is exported: define it with define/who")))

(define parameterized-syntax-class
  '((require (for-syntax racket/base
                         syntax/parse/pre)
             syntax/parse/define)
    (provide m)
    (begin-for-syntax
      (define (small? n limit)
        (< (syntax-e n) limit))
      (define-splicing-syntax-class (below limit)
                                    (pattern (~seq n:nat)
                                      #:when (small? #'n limit))))
    (define-syntax-parse-rule (m (~var n (below 3)))
      n)))

(test-case "a splicing syntax class with parameters is bound like one without"
  (check-equal? (problems-in parameterized-syntax-class) '()))
