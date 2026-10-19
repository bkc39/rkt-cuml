#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre
                              ~and
                              ~not
                              ~optional
                              ~seq
                              define-splicing-syntax-class
                              expr
                              id
                              keyword
                              pattern))
         (only-in cuml/core exn:fail:cuml)
         (only-in racket/string string-append*)
         (only-in raft current-device-resources)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide call-with-params
         define-constructor
         logic-error
         params->string
         resources-of
         updated-params)

(define (logic-error message . args)
  (raise (exn:fail:cuml (apply format message args) (current-continuation-marks) 'logic)))

(define (resources-of params)
  (or (hash-ref params 'resources) (current-device-resources)))

(define (updated-params name params changes)
  (for/fold ([merged params]) ([(key value) (in-hash changes)])
    (cond
      [(hash-has-key? merged key) (hash-set merged key value)]
      [else (logic-error "estimator-set-params: ~a has no parameter ~s" name key)])))

(define (params->string name names params defaults fitted?)
  (define shown
    (for/list ([key (in-list names)]
               #:unless (equal? (hash-ref params key) (hash-ref defaults key)))
      (format " #:~a ~v" key (hash-ref params key))))
  (format "#<~a~a~a>" name (if fitted? " fitted" "") (string-append* shown)))

(define (name->keyword name)
  (string->keyword (symbol->string name)))

(define (call-with-params proc who X params extra)
  (define arguments
    (for/fold ([merged (hash-set params 'resources (resources-of params))])
              ([(key value) (in-hash extra)])
      (hash-set merged key value)))
  (define names (sort (hash-keys arguments) symbol<?))
  (keyword-apply proc
                 (map name->keyword names)
                 (map (lambda (name) (hash-ref arguments name)) names)
                 (list who X)))

(begin-for-syntax
  (define-splicing-syntax-class
   parameter
   #:attributes (keyword name default)
   (pattern (~seq keyword:keyword (~optional (~and default:expr (~not _:keyword))))
     #:with name (format-id #'keyword "~a" (keyword->string (syntax-e #'keyword))))))

(define-syntax-parse-rule (define-constructor (constructor:id p:parameter ...)
                                              #:names names:id
                                              #:defaults defaults:expr
                                              struct-constructor:id)
  (begin
    (define names '(p.name ...))
    (define (constructor (~@ p.keyword [p.name (~? p.default (hash-ref defaults 'p.name))]) ...)
      (struct-constructor (hasheq (~@ 'p.name p.name) ...) #f))))
