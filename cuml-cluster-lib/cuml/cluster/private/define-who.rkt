#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     ;; whole-module: its syntax classes come with it
                     syntax/parse/pre)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "hyperparameters.rkt" absent check-fixed given-hash table-id))

(provide define/who)

(begin-for-syntax
  (define (key->keyword key)
    (string->keyword (symbol->string key)))

  (define ((resolved resolve defaults params keys fixed bound) who body)
    #`(let ([set (#,resolve #,who #,params (given-hash '#,keys (list #,@keys)))])
        #,@(cond
             [(null? fixed) '()]
             [else (list #`(check-fixed #,who set (#,defaults set) '#,fixed))])
        (let* (#,@(for/list ([key (in-list bound)])
                    #`[#,key (hash-ref set '#,key)]))
          #,body)))

  (define-splicing-syntax-class formal
                                #:attributes ([spec 1] [argument 1] wrap)
                                (pattern (~seq #:hyperparameters
                                               table:table-id
                                               (~optional (~seq #:ignoring (ignored:id ...))))
                                  #:do [(define ignored-ids (or (attribute ignored) '()))
                                        (define skipped (map syntax-e ignored-ids))
                                        (define unknown
                                          (for/first ([id (in-list ignored-ids)]
                                                      #:unless (memq (syntax-e id) (attribute table.keys)))
                                            id))]
                                  #:fail-when unknown
                                              (format "~a has no key ~a" (syntax-e #'table) (and unknown (syntax-e unknown)))
                                  #:do [(define (named keys)
                                          (for/list ([key (in-list keys)])
                                            (datum->syntax #'table key #'table)))
                                        (define bound
                                          (for/list ([key (in-list (attribute table.keys))]
                                                     #:unless (memq key skipped))
                                            key))
                                        (define given
                                          (for/list ([key (in-list bound)]
                                                     #:unless (memq key (attribute table.fixed)))
                                            key))]
                                  #:with params (car (generate-temporaries '(params)))
                                  #:with (key ...) (named given)
                                  #:with (keyword ...) (map key->keyword given)
                                  #:with (spec ...) #'((~@ keyword [key absent]) ... #:params [params #f])
                                  #:with (argument ...) #'((~@ keyword key) ... #:params params)
                                  #:attr wrap (resolved #'table.resolve
                                                        #'table.defaults
                                                        #'params
                                                        (named given)
                                                        (for/list ([key (in-list bound)]
                                                                   #:when (memq key (attribute table.fixed)))
                                                          key)
                                                        (named bound)))
                                (pattern id:id
                                  #:with (spec ...) #'(id)
                                  #:with (argument ...) #'(id)
                                  #:attr wrap #f)
                                (pattern (~seq kw:keyword [id:id default:expr])
                                  #:with (spec ...) #'(kw [id default])
                                  #:with (argument ...) #'(kw id)
                                  #:attr wrap #f)))

(define-syntax-parse-rule (define/who (name:id who:id f:formal ...)
                            body:expr ...+)
  #:with name/who (format-id #'name "~a/who" #'name #:source #'name)
  #:with wrapped (for/fold ([body #'(let ()
                                      body ...)])
                           ([wrap (in-list (attribute f.wrap))]
                            #:when wrap)
                   (wrap #'who body))
  (begin
    (define (name/who who f.spec ... ...)
      wrapped)
    (define (name f.spec ... ...)
      (name/who 'name f.argument ... ...))))
