#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     ;; whole-module: its syntax classes come with it
                     syntax/parse/pre)
         (only-in cuml/cluster/private/define-who define-checked)
         (only-in cuml/cluster/private/hyperparameters absent given-hash logic-error table-id)
         (only-in cuml/estimator gen:clusterer gen:estimator gen:predictor gen:scorer gen:transformer)
         (only-in racket/string string-append*)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define-estimator
         logic-error
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

(define (call-fit proc who owner X hyperparameters settings weighted? sample-weight)
  (cond
    [(and sample-weight (not weighted?)) (logic-error "~a: ~a takes no sample-weight" who owner)]
    [weighted?
     (keyword-call proc
                   (list who X)
                   (list* (cons '#:params hyperparameters)
                          (cons '#:sample-weight sample-weight)
                          (setting-arguments settings)))]
    [else
     (keyword-call proc
                   (list who X)
                   (cons (cons '#:params hyperparameters) (setting-arguments settings)))]))

(define (call-applied proc who argument X settings . weight)
  (keyword-call proc
                (list who argument X)
                (append (for/list ([w (in-list weight)])
                          (cons '#:sample-weight w))
                        (setting-arguments settings))))

(begin-for-syntax
  (define-syntax-class applied
    #:attributes (generic [method 1])
    #:datum-literals (predict score transform)
    (pattern [predict proc:expr]
      #:with generic #'gen:predictor
      #:with (method ...) #'((define (predict e X)
                               (call-applied proc 'predict (applied-to 'predict e) X (settings-of e)))))
    (pattern [transform proc:expr]
      #:with generic #'gen:transformer
      #:with (method ...) #'((define (transform e X)
                               (call-applied proc 'transform (applied-to 'transform e) X (settings-of e)))
                             (define (fit-transform e X [_y #f] #:sample-weight [w #f])
                               (call-applied proc
                                             'fit-transform
                                             (argument-of (fitted 'fit-transform e X w))
                                             X
                                             (settings-of e)))))
    (pattern [score proc:expr (~optional (~and weighted #:sample-weight))]
      #:with generic #'gen:scorer
      #:with (method ...) (cond
                            [(attribute weighted)
                             #'((define (score e X [_y #f] #:sample-weight [w #f])
                                  (call-applied proc 'score (applied-to 'score e) X (settings-of e) w)))]
                            [else
                             #'((define (score e X [_y #f] #:sample-weight [_w #f])
                                  (call-applied proc 'score (applied-to 'score e) X (settings-of e))))]))))

(define-syntax-parse-rule (define-estimator name:id
                            #:hyperparameters table:table-id
                            #:fit fit/who:id
                            #:labels labels:expr
                            #:settings (setting:id ...)
                            (~optional (~and no-weight #:no-sample-weight))
                            (~optional (~seq #:apply model->argument:expr entry:applied ...+))
                            (~seq #:methods generic:id methods) ...)
  #:with name* (format-id #'name "~a*" #'name)
  #:with name/proc (format-id #'name "~a/proc" #'name)
  #:with (field ...) (for/list ([field (in-list '(hyperparameters settings model))])
                       (datum->syntax #'name field #'name))
  #:with hyperparameters-of (format-id #'name "~a-hyperparameters" #'name)
  #:with settings-field (format-id #'name "~a-settings" #'name)
  #:with model (format-id #'name "~a-model" #'name)
  #:with (key ...) (generate-temporaries (attribute table.keys))
  #:with (key-name ...) (attribute table.keys)
  #:with (keyword ...) (map (lambda (key) (string->keyword (symbol->string key))) (attribute table.keys))
  #:with (setting-keyword ...) (map (lambda (setting)
                                      (string->keyword (symbol->string (syntax-e setting))))
                                    (attribute setting))
  #:with weighted? (cond
                     [(attribute no-weight) #'#f]
                     [else #'#t])
  (begin
    (define argument-of (~? model->argument values))
    (define (settings-of e)
      (settings-field e))
    (define (fitted who e X sample-weight)
      (call-fit fit/who who 'name X (hyperparameters-of e) (settings-of e) weighted? sample-weight))
    (define (applied-to who e)
      (argument-of (or (model e) (logic-error "~a: ~a is not fitted" who 'name))))
    (struct name (field ...)
      #:constructor-name name*
      #:omit-define-syntaxes
      #:methods gen:estimator
      [(define (fit e X [_y #f] #:sample-weight [w #f])
         (name* (hyperparameters-of e) (settings-of e) (fitted 'fit e X w)))
       (define (fitted? e)
         (and (model e) #t))
       (define (estimator-params e)
         (hyperparameters-of e))
       (define (estimator-set-params e changes)
         (name* (table.update 'estimator-set-params 'name (hyperparameters-of e) changes)
                (settings-of e)
                #f))]
      #:methods gen:clusterer
      [(define (fit-predict e X [_y #f] #:sample-weight [w #f])
         (labels (fitted 'fit-predict e X w)))]
      (~? (~@ (~@ #:methods entry.generic [entry.method ...]) ...))
      (~@ #:methods generic methods) ...
      #:methods gen:equal+hash
      [(define (equal-proc a b recur)
         (and (recur (hyperparameters-of a) (hyperparameters-of b))
              (recur (settings-of a) (settings-of b))
              (eq? (model a) (model b))))
       (define (hash-proc e recur)
         (+ (recur (hyperparameters-of e)) (recur (settings-of e)) (eq-hash-code (model e))))
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
                                       (model e))
                       port))])
    (define-checked #t
                    name
                    name/proc
                    (keyword ... setting-keyword ...)
                    (define (name [params #f]
                                  (~@ keyword [key absent]) ...
                                  (~@ setting-keyword [setting #f]) ...)
                      (name* (table.resolve 'name params (given-hash '(key-name ...) (list key ...)))
                             (hasheq (~@ 'setting setting) ...)
                             #f)))))
