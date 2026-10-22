#lang racket/base

(require (only-in racket/list append-map)
         (only-in racket/syntax format-id)
         (only-in review/ext pop-scope push-scope recur track-binding track-error)
         (only-in syntax/modread with-module-reading-parameterization)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         ;; whole-module: its syntax classes come with it
         syntax/parse/pre)

(provide review-syntax
         should-review?)

(define-syntax-parse-rule (with-scope body:expr ...+)
  (dynamic-wind push-scope
                (lambda ()
                  body ...)
                pop-scope))

(define (review-scoped-body leading body) ;; noqa
  (for-each recur leading)
  (with-scope (for-each recur body)))

(define-syntax-class generic-method
  (pattern (name:id . _)))

(define-syntax-class hyperparameter
  (pattern [key:id (~optional default:expr) (~optional #:fixed)]))

(define-splicing-syntax-class formal
                              #:attributes ([kept 1] table)
                              (pattern (~seq #:hyperparameters table:id
                                             (~optional (~seq #:ignoring (_:id ...))))
                                #:with (kept ...) #'())
                              (pattern one
                                #:with (kept ...) #'(one)
                                #:attr table #f))

(define-syntax-class formal-list
  #:attributes ([kept 1] [table 1])
  (pattern (f:formal ...)
    #:with (kept ...) #'(f.kept ... ...)
    #:with (table ...) (filter values (attribute f.table))))

(define-splicing-syntax-class relay-formal
                              #:attributes ([argument 1])
                              (pattern name:id
                                #:with (argument ...) #'(name))
                              (pattern (~seq kw:keyword [name:id _:expr])
                                #:with (argument ...) #'(kw name)))

(define-syntax-class who-wrapper
  #:datum-literals (define quote)
  (pattern (define (name:id . (~and formals (formal:relay-formal ...)))
             (~and call (relay:id (quote quoted:id) argument ...)))
    #:when (eq? (syntax-e #'quoted) (syntax-e #'name))
    #:when (equal? (format "~a/who" (syntax-e #'name)) (symbol->string (syntax-e #'relay)))
    #:when (equal? (syntax->datum #'(argument ...)) (syntax->datum #'(formal.argument ... ...)))
    #:with procedure #'(lambda formals call)))

(define-syntax-class module-definition
  #:datum-literals (define)
  #:attributes (name)
  (pattern (define (name:id . _)
             . _))
  (pattern (define name:id
             . _)))

(define-syntax-class who-relay
  #:datum-literals (define who)
  (pattern (define (name:id (~and who-id who) . rest)
             body ...+)
    #:with procedure #'(lambda (who-id . rest)
                         body ...)))

(define-syntax-class reviewed-form
  #:datum-literals (define-definition-context-refactoring-rule define-estimator
                                                               define-generics
                                                               define-refactoring-rule
                                                               define-hyperparameters
                                                               define-splicing-syntax-class
                                                               define-syntax-class
                                                               define-syntax-parse-rule
                                                               define/who
                                                               quasiquote
                                                               quote
                                                               struct
                                                               test-gpu
                                                               test-twin
                                                               test-unless-skipped
                                                               test-without-gpu)
  (pattern (quote (_ . _))
    #:attr review void)
  (pattern (quasiquote datum)
    #:attr review (lambda () (for-each recur (unquoted #'datum 1))))
  (pattern (define-syntax-parse-rule (name:id . _)
             body ...)
    #:attr review (lambda ()
                    (track-binding #'name)
                    (mark-uses #'(body ...))))
  (pattern ((~or* define-splicing-syntax-class define-syntax-class) (~or* name:id (name:id . _))
                                                                    . body)
    #:attr review (lambda ()
                    (track-binding #'name #:check-usages? #f)
                    (mark-uses #'body)))
  (pattern ((~or* define-definition-context-refactoring-rule define-refactoring-rule) name:id . body)
    #:attr review (lambda ()
                    (track-binding #'name)
                    (mark-uses #'body)))
  (pattern (define-hyperparameters name:id
             entry:hyperparameter ...)
    #:attr review (lambda ()
                    (track-binding #'name)
                    (for-each recur (filter values (attribute entry.default)))))
  (pattern (define-estimator name:id
             . options)
    #:attr review (lambda ()
                    (track-binding #'name)
                    (track-binding #'name "~a?" #:check-usages? #f)
                    (track-binding #'name "~a-model" #:check-usages? #f)
                    (mark-uses #'options)))

  (pattern (define-generics name:id method:generic-method ... . _)
    #:attr review (lambda ()
                    (track-binding #'name "gen:~a")
                    (track-binding #'name "~a?")
                    (for-each track-binding (attribute method.name))))
  (pattern (struct name:id (~optional super:id)
             fields
             (~alt (~once (~seq #:constructor-name constructor:id))
                   (~once #:omit-define-syntaxes)
                   option) ...)
    #:attr review (lambda () (recur #'(struct name (~? super) fields #:name constructor option ...))))
  (pattern ((~and head define/who) (name:id . formals:formal-list) body ...+)
    #:with name/who (format-id #'name "~a/who" #'name #:source #'name)
    #:attr review (lambda ()
                    (recur #'head)
                    (for-each recur (attribute formals.table))
                    (track-binding #'name #:check-usages? #f)
                    (track-binding #'name/who #:check-usages? #f)
                    (recur #'(lambda (formals.kept ...)
                               body ...))))
  (pattern (test-unless-skipped reason name body ...)
    #:attr review (lambda () (review-scoped-body (list #'reason #'name) (attribute body))))
  (pattern ((~or* test-gpu test-twin test-without-gpu) name body ...)
    #:attr review (lambda () (review-scoped-body (list #'name) (attribute body)))))

(define (should-review? stx)
  (syntax-parse stx
    [relay:who-relay (exported? #'relay.name)]
    [_:who-wrapper #t]
    [_:reviewed-form #t]
    [_ #f]))

(define (review-syntax stx)
  (syntax-parse stx
    [relay:who-relay
     (track-error #'relay.name
                  (format "~a takes who first and is exported: define it with define/who"
                          (syntax-e #'relay.name)))
     (track-binding #'relay.name)
     (recur #'relay.procedure)]
    [wrapper:who-wrapper
     (track-error #'wrapper.name
                  (format "~a only relays to ~a: define both with define/who"
                          (syntax-e #'wrapper.name)
                          (syntax-e #'wrapper.relay)))
     (track-binding #'wrapper.name)
     (recur #'wrapper.procedure)]
    [form:reviewed-form ((attribute form.review))]))

(define (mark-uses stx)
  (syntax-parse stx
    [x:id (recur #'x)]
    [(a . d)
     (mark-uses #'a)
     (mark-uses #'d)]
    [#(e ...) (for-each mark-uses (attribute e))]
    [_ (void)]))

(define provides-by-file (make-hash))

(define (exported? name)
  (and (memq (syntax-e name)
             (hash-ref! provides-by-file
                        (syntax-source name)
                        (lambda () (provided-names (syntax-source name)))))
       (not (regexp-match? #rx"/who$" (symbol->string (syntax-e name))))))

(define (provided-names source)
  (cond
    [(path-string? source)
     (syntax-parse (with-input-from-file source
                                         (lambda ()
                                           (port-count-lines! (current-input-port))
                                           (with-module-reading-parameterization read-syntax)))
       [(_module _name _lang (_begin form ...))
        (define defined (append-map defined-in (attribute form)))
        (append-map (lambda (one) (provided-in one defined)) (attribute form))]
       [_ '()])]
    [else '()]))

(define (defined-in form)
  (syntax-parse form
    [definition:module-definition (list (syntax-e #'definition.name))]
    [_ '()]))

(define (provided-in form defined)
  (syntax-parse form
    #:datum-literals (provide)
    [(provide spec ...) (append-map (lambda (one) (spec-names one defined)) (attribute spec))]
    [_ '()]))

(define (spec-names spec defined)
  (syntax-parse spec
    #:datum-literals (all-defined-out combine-out except-out prefix-out rename-out)
    [name:id (list (syntax-e #'name))]
    [(rename-out [inner:id _] ...) (map syntax-e (attribute inner))]
    [(prefix-out _ inner ...) (append-map (lambda (one) (spec-names one defined)) (attribute inner))]
    [(combine-out inner ...) (append-map (lambda (one) (spec-names one defined)) (attribute inner))]
    [(except-out inner removed ...)
     (remq* (append-map (lambda (one) (spec-names one defined)) (attribute removed))
            (spec-names #'inner defined))]
    [(all-defined-out) defined]
    [_ '()]))

(define (unquoted stx depth)
  (syntax-parse stx
    #:datum-literals (quasiquote unquote unquote-splicing)
    [((~or* unquote unquote-splicing) e)
     #:when (= depth 1)
     (list #'e)]
    [((~or* unquote unquote-splicing) e) (unquoted #'e (sub1 depth))]
    [(quasiquote e) (unquoted #'e (add1 depth))]
    [(a . d) (append (unquoted #'a depth) (unquoted #'d depth))]
    [#(e ...) (append-map (lambda (e) (unquoted e depth)) (syntax->list #'(e ...)))]
    [_ (append-map (lambda (e) (unquoted e depth)) (quasiquoted-parts (syntax-e stx)))]))

(define (quasiquoted-parts datum)
  (cond
    [(box? datum) (list (unbox datum))]
    [(hash? datum) (hash-map datum (lambda (_key value) value) #t)]
    [(prefab-struct-key datum) (cdr (vector->list (struct->vector datum)))]
    [else '()]))
