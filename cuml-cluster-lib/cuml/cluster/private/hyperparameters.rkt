#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/match match)
                     (only-in racket/syntax format-id)
                     syntax/parse/pre)
         (only-in cuml/core exn:fail:cuml)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide (for-syntax key->keyword
                     keyword-checked
                     mention
                     proc-name
                     table-id)
         absent
         define-hyperparameters
         given-hash
         logic-error)

(define absent (string->uninterned-symbol "absent"))

(define (logic-error message . args)
  (raise (exn:fail:cuml (apply format message args) (current-continuation-marks) 'logic)))

(define (given-hash keys arguments)
  (for/hasheq ([key (in-list keys)]
               [value (in-list arguments)]
               #:unless (eq? value absent))
    (values key value)))

(define (printed key)
  (format "~s" key))

(define (refuse-unknown who owner keys set)
  (match (sort (for/list ([key (in-hash-keys set)]
                          #:unless (memq key keys))
                 key)
               string<?
               #:key printed)
    [(cons key _) (logic-error "~a: ~a has no parameter ~s" who owner key)]
    ['() (void)]))

(define (refuse-incomplete who owner keys set)
  (refuse-unknown who owner keys set)
  (match (for/first ([key (in-list keys)]
                     #:unless (hash-has-key? set key))
           key)
    [#f (void)]
    [key (logic-error "~a: expected every ~a parameter, missing ~s" who owner key)]))

(define (layered keys base changes)
  (for/hasheq ([key (in-list keys)])
    (values key (hash-ref changes key (lambda () (hash-ref base key))))))

(define (resolve-set who owner keys fill base given)
  (cond
    [(not base) (fill given)]
    [(hash? base)
     (refuse-incomplete who owner keys base)
     (layered keys base given)]
    [else (raise-argument-error who "(or/c #f hash?)" base)]))

(define (default-or-absent default)
  (with-handlers ([exn:fail? (lambda (_) absent)])
    (default)))

(define (update-set who owner keys base changes)
  (refuse-unknown who owner keys changes)
  (refuse-incomplete who owner keys base)
  (layered keys base changes))

(begin-for-syntax
  (define (key->keyword key)
    (string->keyword (symbol->string key)))

  (define (proc-name id)
    (format-id id "~a/proc" id #:source id))

  (define params-suffix #rx"(.)-params$")

  (define (owner-of table)
    (string->symbol (regexp-replace params-suffix (symbol->string (syntax-e table)) "\\1")))

  (struct keyword-checked (procedure keywords)
    #:property prop:procedure
    (lambda (checked stx)
      (define procedure (keyword-checked-procedure checked))
      (syntax-parse stx
        [_:id procedure]
        [(head . arguments)
         (match (unknown-keyword (syntax->list #'arguments) (keyword-checked-keywords checked))
           [#f (datum->syntax stx (cons procedure #'arguments) stx stx)]
           [kw
            (raise-syntax-error (syntax-e #'head)
                                (format "does not take the keyword ~a" (syntax-e kw))
                                stx
                                kw)])])))

  (define (unknown-keyword arguments keywords)
    (match arguments
      [(list* (and kw-stx (app syntax-e (? keyword? kw))) _ rest)
       (cond
         [(memq kw keywords) (unknown-keyword rest keywords)]
         [else kw-stx])]
      [(cons _ rest) (unknown-keyword rest keywords)]
      [_ #f]))

  (struct hyperparameter-table keyword-checked (keys fixed resolve defaults update))

  (define-syntax-class table-id
    #:description "a hyperparameter table"
    #:attributes (owner keys fixed resolve defaults update)
    (pattern id:id
      #:attr table (syntax-local-value #'id (lambda () #f))
      #:fail-unless (hyperparameter-table?
                     (attribute table)) "expected a table from define-hyperparameters"
      #:attr owner (datum->syntax #'id (owner-of #'id))
      #:attr keys (hyperparameter-table-keys (attribute table))
      #:attr fixed (hyperparameter-table-fixed (attribute table))
      #:attr resolve (hyperparameter-table-resolve (attribute table))
      #:attr defaults (hyperparameter-table-defaults (attribute table))
      #:attr update (hyperparameter-table-update (attribute table))))

  (define-syntax-class params-name
    #:description "a table name ending in -params"
    (pattern name:id
      #:fail-unless
      (regexp-match? params-suffix (symbol->string (syntax-e #'name)))
      "expected a table name ending in -params"))

  (define-syntax-class hyperparameter
    #:description "a hyperparameter [key default] or [key default #:fixed]"
    #:attributes (key default fixed? keyword)
    (pattern [key:id #:fixed]
      #:fail-when #t "a #:fixed key needs a default"
      #:attr default #f
      #:attr fixed? #f
      #:attr keyword #f)
    (pattern [key:id default:expr (~optional (~and fixed #:fixed))]
      #:fail-when (and (eq? (syntax-e #'key) 'params)
                       #'key) "params is the #:params keyword of every entry"
      #:attr fixed? (and (attribute fixed) #t)
      #:with keyword (key->keyword (syntax-e #'key))))

  (define (mention stx ids)
    (syntax-parse stx
      #:literals (quote)
      [(quote _) #f]
      [x:id
       (for/first ([id (in-list ids)]
                   #:when (bound-identifier=? #'x id))
         #'x)]
      [(a . d) (or (mention #'a ids) (mention #'d ids))]
      [_ #f]))

  (define (forward-reference keys defaults)
    (for/or ([default (in-list defaults)]
             [position (in-naturals)])
      (mention default (list-tail keys position))))

  (define-splicing-syntax-class
   hyperparameter-list
   #:attributes ([key 1] [default 1] [keyword 1] fixed-keys)
   (pattern (~seq entry:hyperparameter ...+)
     #:fail-when (check-duplicate-identifier (attribute entry.key)) "duplicate hyperparameter"
     #:fail-when
     (forward-reference (attribute entry.key) (attribute entry.default))
     "a default refers to its own or a later key"
     #:with (key ...) #'(entry.key ...)
     #:with (default ...) #'(entry.default ...)
     #:with (keyword ...) #'(entry.keyword ...)
     #:attr fixed-keys (for/list ([key (in-list (attribute entry.key))]
                                  [fixed? (in-list (attribute entry.fixed?))]
                                  #:when fixed?)
                         (syntax-e key)))))

(define-syntax-parse-rule (define-hyperparameters table:params-name entries:hyperparameter-list)
  #:with proc (proc-name #'table)
  #:with owner (owner-of #'table)
  #:with (fixed-key ...) (attribute entries.fixed-keys)
  #:with (default-value ...) (generate-temporaries #'(entries.key ...))
  (begin
    (define keys '(entries.key ...))
    (define (fill given)
      (let* ([entries.key (hash-ref given 'entries.key (lambda () entries.default))] ...)
        (hasheq (~@ 'entries.key entries.key) ...)))
    (define (defaults-for set)
      (let* ([~@
              [default-value (default-or-absent (lambda () entries.default))]
              [entries.key (hash-ref set 'entries.key)]] ...)
        (hasheq (~@ 'entries.key default-value) ...)))
    (define (resolve who owner base given)
      (resolve-set who owner keys fill base given))
    (define (update who owner base changes)
      (update-set who owner keys base changes))
    (define proc
      (let ()
        (define (table [base #f] (~@ entries.keyword [entries.key absent]) ...)
          (resolve 'table 'owner base (given-hash keys (list entries.key ...))))
        table))
    (define-syntax table
      (hyperparameter-table (quote-syntax proc)
                            '(entries.keyword ...)
                            '(entries.key ...)
                            '(fixed-key ...)
                            (quote-syntax resolve)
                            (quote-syntax defaults-for)
                            (quote-syntax update)))))
