#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     syntax/parse/pre)
         (only-in cuml/core/private/define-who define-checked)
         (only-in cuml/core/private/hyperparameters absent given-hash key->keyword proc-name table-id)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in cuml/estimator gen:clusterer gen:estimator gen:predictor gen:scorer gen:transformer)
         (only-in racket/string string-append*)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define-estimator
         params->string)

(define (params->string name keys hyperparameters defaults setting-names settings fitted?)
  (define shown
    (append (for/list ([key (in-list keys)]
                       #:unless (equal? (hash-ref hyperparameters key) (hash-ref defaults key)))
              (format " #:~a ~v" key (hash-ref hyperparameters key)))
            (for/list ([setting (in-list setting-names)]
                       #:when (hash-ref settings setting))
              (format " #:~a ~v" setting (hash-ref settings setting)))))
  (format "#<~a~a~a>" name (if fitted? " fitted" "") (string-append* shown)))

(define (symbol->keyword name)
  (string->keyword (symbol->string name)))

(define (setting-arguments settings)
  (for/list ([(setting value) (in-hash settings)]
             #:when value)
    (cons (symbol->keyword setting) value)))

(define (keyword-call proc positional named)
  (define sorted (sort named keyword<? #:key car))
  (keyword-apply proc (map car sorted) (map cdr sorted) positional))

(define (refuse-weight who owner sample-weight)
  (cond
    [sample-weight (logic-error "~a: ~a takes no sample-weight" who owner)]
    [else absent]))

(define (call-fit proc who owner X hyperparameters settings weighted? sample-weight)
  (define named (cons (cons '#:params hyperparameters) (setting-arguments settings)))
  (cond
    [weighted? (keyword-call proc (list who X) (cons (cons '#:sample-weight sample-weight) named))]
    [else
     (refuse-weight who owner sample-weight)
     (keyword-call proc (list who X) named)]))

(define (call-applied proc who argument X settings [sample-weight absent])
  (define named (setting-arguments settings))
  (keyword-call proc
                (list who argument X)
                (cond
                  [(eq? sample-weight absent) named]
                  [else (cons (cons '#:sample-weight sample-weight) named)])))

(begin-for-syntax
  (define-syntax-class applied
    #:attributes ([clause 1])
    #:datum-literals (predict score transform)
    (pattern [predict proc:expr]
      #:with (clause ...) #'(#:methods gen:predictor
                                       [(define (predict e X)
                                          (on-model proc 'predict e X))]))
    (pattern [transform proc:expr]
      #:with (clause ...) #'(#:methods gen:transformer
                                       [(define (transform e X)
                                          (on-model proc 'transform e X))
                                        (define (fit-transform e X [_y #f] #:sample-weight [w #f])
                                          (on-fit proc 'fit-transform e X w))]))
    (pattern [score proc:expr (~optional (~and weighted #:sample-weight))]
      #:attr weight (and (attribute weighted) #'w)
      #:with (clause ...) #'(#:methods gen:scorer
                             [(define (score e X [_y #f] #:sample-weight [w #f])
                                (on-model proc 'score e X (~? weight (unweighted 'score w))))])))

  (define-splicing-syntax-class method-clause
                                #:attributes ([clause 1])
                                (pattern (~seq #:methods generic:id methods)
                                  #:with (clause ...) #'(#:methods generic methods)))

  (define-syntax-class (setting-name keys)
    #:description "a setting"
    (pattern name:id
      #:fail-when
      (memq (syntax-e #'name) (cons 'params keys))
      (format "the setting ~a is also a hyperparameter or #:params" (syntax-e #'name)))))

(define-syntax-parse-rule (define-estimator name:id
                            #:hyperparameters table:table-id
                            #:fit fit/who:id
                            #:labels labels:expr
                            #:settings ((~var setting (setting-name (attribute table.keys))) ...)
                            (~optional (~and no-weight #:no-sample-weight))
                            (~optional (~seq #:apply model->argument:expr entry:applied ...+))
                            extra:method-clause ...)
  #:with name* (format-id #'name "~a*" #'name)
  #:with (field ...) (for/list ([field (in-list '(hyperparameters settings model))])
                       (datum->syntax #'name field #'name))
  #:with (hyperparameters-of settings-of
                             model-of) (for/list ([field (in-list (syntax->datum #'(field ...)))])
                                         (format-id #'name "~a-~a" #'name field))
  #:with (key ...) (generate-temporaries (attribute table.keys))
  #:with (key-name ...) (attribute table.keys)
  #:with (keyword ...) (map key->keyword (attribute table.keys))
  #:with (setting-keyword ...) (map key->keyword (syntax->datum #'(setting ...)))
  #:with weighted? (cond
                     [(attribute no-weight) #'#f]
                     [else #'#t])
  (begin
    (define argument-of (~? model->argument values))
    (define (fitted who e X sample-weight)
      (call-fit fit/who who 'name X (hyperparameters-of e) (settings-of e) weighted? sample-weight))
    (define (on-model proc who e X [sample-weight absent])
      (call-applied proc
                    who
                    (argument-of (or (model-of e) (logic-error "~a: ~a is not fitted" who 'name)))
                    X
                    (settings-of e)
                    sample-weight))
    (define (on-fit proc who e X sample-weight)
      (call-applied proc who (argument-of (fitted who e X sample-weight)) X (settings-of e)))
    (define (unweighted who sample-weight)
      (refuse-weight who 'name sample-weight))
    (struct name (field ...)
      #:constructor-name name*
      #:omit-define-syntaxes
      #:methods gen:estimator
      [(define (fit e X [_y #f] #:sample-weight [w #f])
         (name* (hyperparameters-of e) (settings-of e) (fitted 'fit e X w)))
       (define (fitted? e)
         (and (model-of e) #t))
       (define (estimator-params e)
         (hyperparameters-of e))
       (define (estimator-set-params e changes)
         (name* (table.update 'estimator-set-params 'name (hyperparameters-of e) changes)
                (settings-of e)
                #f))]
      #:methods gen:clusterer
      [(define (fit-predict e X [_y #f] #:sample-weight [w #f])
         (labels (fitted 'fit-predict e X w)))]
      (~? (~@ entry.clause ... ...))
      extra.clause ...
      ...
      #:methods gen:equal+hash
      [(define (equal-proc a b recur)
         (and (recur (hyperparameters-of a) (hyperparameters-of b))
              (recur (settings-of a) (settings-of b))
              (eq? (model-of a) (model-of b))))
       (define (hash-proc e recur)
         (+ (recur (hyperparameters-of e)) (recur (settings-of e)) (eq-hash-code (model-of e))))
       (define (hash2-proc e recur)
         (+ (recur (hyperparameters-of e)) (recur (settings-of e))))]
      #:methods gen:custom-write
      [(define (write-proc e port _mode)
         (write-string (params->string 'name
                                       '(key-name ...)
                                       (hyperparameters-of e)
                                       (table.defaults (hyperparameters-of e))
                                       '(setting ...)
                                       (settings-of e)
                                       (model-of e))
                       port))])
    (define-checked
     name
     (keyword ... setting-keyword ...)
     (define (name [params #f] (~@ keyword [key absent]) ... (~@ setting-keyword [setting #f]) ...)
       (name* (table.resolve 'name 'name params (given-hash '(key-name ...) (list key ...)))
              (hasheq (~@ 'setting setting) ...)
              #f)))))
