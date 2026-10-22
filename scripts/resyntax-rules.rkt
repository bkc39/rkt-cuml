#lang racket/base

(require (only-in resyntax/base
                  define-definition-context-refactoring-rule
                  define-refactoring-suite
                  ~focus-replacement-on
                  ~replacement)
         (only-in resyntax/default-recommendations default-recommendations)
         (only-in syntax/parse/pre ...+
                  define-splicing-syntax-class
                  expr
                  id
                  keyword
                  pattern
                  ~and
                  ~seq))

(provide cuml-recommendations
         cuml-rules
         relay-formal)

(define-splicing-syntax-class relay-formal
                              #:attributes ([argument 1])
                              (pattern name:id
                                #:with (argument ...) #'(name))
                              (pattern (~seq kw:keyword [name:id _:expr])
                                #:with (argument ...) #'(kw name)))

(define (relay-of? name relay)
  (equal? (format "~a/who" (syntax-e name)) (symbol->string (syntax-e relay))))

(define-definition-context-refactoring-rule
 who-pair-to-define/who
 #:description "This procedure and its /who relay are a define/who."
 #:literals (define quote)
 (~seq body-before ...
       (~and relay
             (define (relay-name:id who:id . (~and formals (formal:relay-formal ...)))
               relay-body ...+))
       (~and wrapper
             (define (name:id wrapper-formal ...)
               (called:id (quote quoted:id) argument ...)))
       body-after ...)
 #:when (relay-of? #'name #'relay-name)
 #:when (free-identifier=? #'called #'relay-name)
 #:when (eq? (syntax-e #'quoted) (syntax-e #'name))
 #:when (equal? (syntax->datum #'(wrapper-formal ...)) (syntax->datum #'formals))
 #:when (equal? (syntax->datum #'(argument ...)) (syntax->datum #'(formal.argument ... ...)))
 #:with define-who
 (syntax-property #'define/who 'skip-incorrect-binding-check? #t)
 #:with replacement
 #'(~replacement (define-who (name who . formals) relay-body ...) #:original-splice (relay wrapper))
 (body-before ... (~focus-replacement-on replacement) body-after ...))

(define-refactoring-suite cuml-rules #:rules (who-pair-to-define/who))

(define-refactoring-suite cuml-recommendations #:suites (default-recommendations cuml-rules))
