#lang racket/base

(require (only-in cuml/cluster/private/kmeans
                  init-rows
                  kmeans-defaults
                  kmeans-fit/who
                  kmeans-predict/who
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-score/who
                  kmeans-transform/who)
         (only-in cuml/core exn:fail:cuml)
         (only-in cuml/estimator gen:clusterer gen:estimator gen:predictor gen:scorer gen:transformer)
         (only-in racket/string string-append*)
         (only-in raft current-device-resources))

(provide kmeans
         kmeans?
         kmeans-model)

(define parameter-names
  '(init n-clusters
         max-iter
         tol
         n-init
         random-state
         oversampling-factor
         max-samples-per-batch
         resources))

(define (name->keyword name)
  (string->keyword (symbol->string name)))

(define (default name)
  (hash-ref kmeans-defaults name))

(define (logic-error message . args)
  (raise (exn:fail:cuml (apply format message args) (current-continuation-marks) 'logic)))

(define (resources-of params)
  (or (hash-ref params 'resources) (current-device-resources)))

(define (fitted-model who e X sample-weight)
  (define params (kmeans-params e))
  (define arguments (hash-set* params 'resources (resources-of params) 'sample-weight sample-weight))
  (define names (sort (hash-keys arguments) symbol<?))
  (keyword-apply kmeans-fit/who
                 (map name->keyword names)
                 (map (lambda (name) (hash-ref arguments name)) names)
                 (list who X)))

(define (model-of who e)
  (or (kmeans-model e) (logic-error "~a: kmeans is not fitted" who)))

(define (on-centroids proc who e model X)
  (define params (kmeans-params e))
  (proc who
        (kmeans-result-centroids model)
        X
        #:max-samples-per-batch (hash-ref params 'max-samples-per-batch)
        #:resources (resources-of params)))

(define (scored who e X sample-weight)
  (define params (kmeans-params e))
  (kmeans-score/who who
                    (kmeans-result-centroids (model-of who e))
                    X
                    #:sample-weight sample-weight
                    #:max-samples-per-batch (hash-ref params 'max-samples-per-batch)
                    #:resources (resources-of params)))

(define (updated params changes)
  (for/fold ([merged params]) ([(key value) (in-hash changes)])
    (cond
      [(hash-has-key? merged key) (hash-set merged key value)]
      [else (logic-error "estimator-set-params: kmeans has no parameter ~s" key)])))

(define (kmeans->string e)
  (define params (kmeans-params e))
  (define defaults
    (hash-set (kmeans-params (kmeans))
              'n-clusters
              (init-rows (hash-ref params 'init) (default 'n-clusters))))
  (define marker (if (kmeans-model e) " fitted" ""))
  (define shown
    (for/list ([name (in-list parameter-names)]
               #:unless (equal? (hash-ref params name) (hash-ref defaults name)))
      (format " #:~a ~v" name (hash-ref params name))))
  (format "#<kmeans~a~a>" marker (string-append* shown)))

(struct kmeans (params model)
  #:constructor-name kmeans*
  #:omit-define-syntaxes
  #:methods gen:estimator
  [(define (fit e X [_y #f] #:sample-weight [sample-weight #f])
     (kmeans* (kmeans-params e) (fitted-model 'fit e X sample-weight)))
   (define (fitted? e)
     (and (kmeans-model e) #t))
   (define (estimator-params e)
     (kmeans-params e))
   (define (estimator-set-params e params)
     (kmeans* (updated (kmeans-params e) params) #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (on-centroids kmeans-predict/who 'predict e (model-of 'predict e) X))]
  #:methods gen:clusterer
  [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
     (kmeans-result-labels (fitted-model 'fit-predict e X sample-weight)))]
  #:methods gen:transformer
  [(define (transform e X)
     (on-centroids kmeans-transform/who 'transform e (model-of 'transform e) X))
   (define (fit-transform e X [_y #f] #:sample-weight [sample-weight #f])
     (on-centroids kmeans-transform/who
                   'fit-transform
                   e
                   (fitted-model 'fit-transform e X sample-weight)
                   X))]
  #:methods gen:scorer
  [(define (score e X [_y #f] #:sample-weight [sample-weight #f])
     (scored 'score e X sample-weight))]
  #:methods gen:equal+hash
  [(define (equal-proc a b recur)
     (and (recur (kmeans-params a) (kmeans-params b)) (eq? (kmeans-model a) (kmeans-model b))))
   (define (hash-proc e recur)
     (+ (recur (kmeans-params e)) (eq-hash-code (kmeans-model e))))
   (define (hash2-proc e recur)
     (recur (kmeans-params e)))]
  #:methods gen:custom-write
  [(define (write-proc e port _mode)
     (write-string (kmeans->string e) port))])

(define (kmeans #:init (init (default 'init))
                #:n-clusters [n-clusters (init-rows init (default 'n-clusters))]
                #:max-iter [max-iter (default 'max-iter)]
                #:tol [tol (default 'tol)]
                #:n-init [n-init (default 'n-init)]
                #:random-state [random-state (default 'random-state)]
                #:oversampling-factor [oversampling-factor (default 'oversampling-factor)]
                #:max-samples-per-batch [max-samples-per-batch (default 'max-samples-per-batch)]
                #:resources [resources #f])
  (kmeans* (for/hasheq ([name (in-list parameter-names)]
                        [value (in-list (list init
                                              n-clusters
                                              max-iter
                                              tol
                                              n-init
                                              random-state
                                              oversampling-factor
                                              max-samples-per-batch
                                              resources))])
             (values name value))
           #f))
