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

(define estimator-and-its-constructor
  '((provide widget
             widget?
             widget-model)
    (define-syntax-parse-rule (define-estimator . _)
      (void))
    (define (fitted who e X weight)
      (list who e X weight))
    (define-estimator widget
                      #:fitted fitted
                      #:labels car
                      #:string (lambda (e) (format "~a" (widget-params e)))
                      #:methods gen:custom-write
                      [(define (write-proc e port _mode)
                         (write (widget-model e) port))])
    (define (widget)
      (widget* (hasheq) #f))))

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

(define page-helpers-and-their-blocks
  '((provide helpers
             helpers-block
             require-block
             required)
    (define-syntax-parse-rule (define-page-helpers name ...)
      (void))
    (define-page-helpers required require-block helpers helpers-block)))

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

(test-case "define-estimator binds the struct with its constructor name, and reads its options"
  (check-equal? (problems-in estimator-and-its-constructor) '()))

(test-case "define-page-helpers binds each of its names"
  (check-equal? (problems-in page-helpers-and-their-blocks) '()))

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
