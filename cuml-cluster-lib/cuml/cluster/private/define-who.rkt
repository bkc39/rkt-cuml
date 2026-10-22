#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     syntax/parse/pre)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "hyperparameters.rkt"
                  absent
                  given-hash
                  key->keyword
                  keyword-checked
                  logic-error
                  mention
                  proc-name
                  table-id))

(provide define-checked
         define/who)

(define (check-fixed who set defaults keys)
  (match (findf (lambda (key)
                  (define default (hash-ref defaults key))
                  (not (or (eq? default absent) (equal? (hash-ref set key) default))))
                keys)
    [#f (void)]
    [key
     (logic-error "~a: ~a: expected ~a, got ~a" who key (hash-ref defaults key) (hash-ref set key))]))

(begin-for-syntax
  (define (named table keys)
    (for/list ([key (in-list keys)])
      (datum->syntax table key table)))

  (define (earlier-key-reference defaults keys)
    (for/fold ([seen '()]
               [found #f]
               #:result found)
              ([default (in-list defaults)]
               [bound (in-list keys)])
      (values (append seen bound) (or found (and default (mention default seen))))))

  (define-syntax-class (table-key table keys)
    #:description "a key of the table"
    (pattern key:id
      #:fail-unless (memq (syntax-e #'key)
                          keys) (format "~a has no key ~a" (syntax-e table) (syntax-e #'key))))

  (define-splicing-syntax-class
   (formal who-id)
   #:attributes ([spec 1] [argument 1] [binding 1] [key 1] default)
   (pattern (~seq #:hyperparameters table:table-id
                  (~optional (~seq #:ignoring ~!
                                   ((~var ignored (table-key #'table (attribute table.keys))) ...))
                             #:defaults ([(ignored 1) '()])))
     #:attr skipped (map syntax-e (attribute ignored))
     #:with (key ...) (named #'table (remq* (attribute skipped) (attribute table.keys)))
     #:with (given-key ...) (remq* (append (attribute skipped) (attribute table.fixed))
                                   (attribute table.keys))
     #:with (fixed-key ...) (remq* (attribute skipped) (attribute table.fixed))
     #:with (given-value ...) (generate-temporaries #'(given-key ...))
     #:with (keyword ...) (map key->keyword (syntax->datum #'(given-key ...)))
     #:with (who params set checked) (cons who-id (generate-temporaries '(params set checked)))
     #:attr check (and (pair? (attribute fixed-key))
                       #'[checked (check-fixed who set (table.defaults set) '(fixed-key ...))])
     #:with (spec ...) #'((~@ keyword [given-value absent]) ... #:params [params #f])
     #:with (argument ...) #'((~@ keyword given-value) ... #:params params)
     #:with
     (binding ...)
     #'([set
         (table.resolve who 'table.owner params (given-hash '(given-key ...) (list given-value ...)))]
        (~? check)
        [key (hash-ref set 'key)] ...)
     #:attr default #f)
   (pattern id:id
     #:with (spec ...) #'(id)
     #:with (argument ...) #'(id)
     #:with (binding ...) #'()
     #:with (key ...) #'()
     #:attr default #f)
   (pattern (~seq kw:keyword [id:id default:expr])
     #:with (spec ...) #'(kw [id default])
     #:with (argument ...) #'(kw id)
     #:with (binding ...) #'()
     #:with (key ...) #'())))

(define-syntax-parse-rule (define-checked name:id (kw:keyword ...) definition)
  #:with name/proc (proc-name #'name)
  (begin
    (define name/proc
      (let ()
        definition
        name))
    (define-syntax name (keyword-checked (quote-syntax name/proc) '(kw ...)))))

(define-syntax-parse-rule (define/who (name:id who:id (~var f (formal #'who)) ...)
                            body:expr ...+)
  #:fail-when
  (earlier-key-reference (attribute f.default) (attribute f.key))
  "a default refers to a hyperparameter of an earlier #:hyperparameters"
  #:with name/who (format-id #'name "~a/who" #'name #:source #'name)
  #:with (kw ...) (filter (lambda (spec) (keyword? (syntax-e spec)))
                          (syntax->list #'(f.spec ... ...)))
  #:with public #'(define (name f.spec ... ...)
                    (name/who 'name f.argument ... ...))
  #:with definition (cond
                      [(null? (syntax->list #'(f.binding ... ...))) #'public]
                      [else #'(define-checked name (kw ...) public)])
  (begin
    (define (name/who who f.spec ... ...)
      (let* (f.binding ...
             ...)
        body ...))
    definition))
