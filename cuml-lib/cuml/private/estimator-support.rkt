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
         (only-in cuml/estimator gen:clusterer gen:estimator)
         (only-in racket/string string-append*)
         (only-in raft current-device-resources)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide call-with-params
         define-constructor
         define-estimator
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

(define-syntax-parse-rule (define-estimator name:id
                                            #:fitted fitted:expr
                                            #:labels labels:expr
                                            #:string ->string:expr
                                            (~seq #:methods generic:id methods) ...)
  #:with constructor (format-id #'name "~a*" #'name)
  #:with params (format-id #'name "~a-params" #'name)
  #:with model (format-id #'name "~a-model" #'name)
  #:with (params-field model-field) (list (format-id #'name "params") (format-id #'name "model"))
  (struct name (params-field model-field)
    #:constructor-name constructor
    #:omit-define-syntaxes
    #:methods gen:estimator
    [(define (fit e X [_y #f] #:sample-weight [sample-weight #f])
       (constructor (params e) (fitted 'fit e X sample-weight)))
     (define (fitted? e)
       (and (model e) #t))
     (define (estimator-params e)
       (params e))
     (define (estimator-set-params e changes)
       (constructor (updated-params 'name (params e) changes) #f))]
    #:methods gen:clusterer
    [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
       (labels (fitted 'fit-predict e X sample-weight)))]
    (~@ #:methods generic methods) ...
    #:methods gen:equal+hash
    [(define (equal-proc a b recur)
       (and (recur (params a) (params b)) (eq? (model a) (model b))))
     (define (hash-proc e recur)
       (+ (recur (params e)) (eq-hash-code (model e))))
     (define (hash2-proc e recur)
       (recur (params e)))]
    #:methods gen:custom-write
    [(define (write-proc e port _mode)
       (write-string (->string e) port))]))
