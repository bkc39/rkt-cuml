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
  (pattern (~seq #:hyperparameters table:id (~optional (~seq #:ignoring (_:id ...))))
    #:with (kept ...) #'())
  (pattern one
    #:with (kept ...) #'(one)
    #:attr table #f))

(define-syntax-class formal-list
  #:attributes ([kept 1] [table 1])
  (pattern (f:formal ...)
    #:with (kept ...) #'(f.kept ... ...)
    #:with (table ...) (filter values (attribute f.table))))

(define-syntax-class who-relay
  (pattern ((~datum define) (name:id (~and who (~datum who)) . rest) body ...+)
    #:with procedure #'(lambda (who . rest)
                         body ...)))

(define-syntax-class reviewed-form
  #:datum-literals (define-estimator define-generics
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
  (pattern ((~or* define-splicing-syntax-class define-syntax-class) name:id . body)
    #:attr review (lambda ()
                    (track-binding #'name #:check-usages? #f)
                    (mark-uses #'body)))
  (pattern (define-hyperparameters name:id entry:hyperparameter ...)
    #:attr review (lambda ()
                    (track-binding #'name)
                    (for-each recur (filter values (attribute entry.default)))))
  (pattern (define-estimator name:id . options)
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
  (and (not (regexp-match? #rx"/who$" (symbol->string (syntax-e name))))
       (memq (syntax-e name) (hash-ref! provides-by-file (syntax-source name) (lambda () (provided-names (syntax-source name)))))
       #t))

(define (provided-names source)
  (cond
    [(path-string? source)
     (syntax-parse (with-input-from-file source
                                         (lambda ()
                                           (port-count-lines! (current-input-port))
                                           (with-module-reading-parameterization read-syntax)))
       [(_module _name _lang (_begin form ...)) (append-map provided-in (attribute form))]
       [_ '()])]
    [else '()]))

(define (provided-in form)
  (syntax-parse form
    #:datum-literals (provide rename-out)
    [(provide spec ...)
     (for*/list ([spec (in-list (attribute spec))]
                 [name (in-list (syntax-parse spec
                                  [name:id (list #'name)]
                                  [(rename-out [inner:id _] ...) (attribute inner)]
                                  [_ '()]))])
       (syntax-e name))]
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
