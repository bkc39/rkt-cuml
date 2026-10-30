#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  dbscan
                  hdbscan
                  kmeans
                  pca
                  spectral-clustering
                  standard-scaler
                  truncated-svd
                  umap)
         (only-in cuml/cluster
                  dbscan-fit
                  dbscan-params
                  hdbscan-fit
                  hdbscan-params
                  kmeans-fit
                  kmeans-params
                  single-linkage-fit
                  single-linkage-params
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-params)
         (only-in cuml/decomposition
                  pca-fit
                  pca-fit-transform
                  pca-params
                  tsvd-fit
                  tsvd-fit-transform
                  tsvd-params)
         (only-in cuml/manifold umap-fit umap-fit-transform umap-params)
         (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-fit-transform
                  standard-scaler-params)
         (only-in racket/dict dict-ref)
         (only-in racket/list append-map takef)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-not-false test-case)
         (only-in scribble/reader read-inside))

(define (manual-forms file)
  (call-with-input-file (collection-file-path file "cuml" "scribblings" "reference")
                        (lambda (in)
                          (read-line in 'any)
                          (read-inside in))))

(define (defprocs datum)
  (match datum
    [(list* 'defproc (cons (? symbol? name) arguments) _) (list (cons name arguments))]
    [(cons head tail) (append (defprocs head) (defprocs tail))]
    [_ '()]))

(define documented
  (append-map defprocs
              (append-map manual-forms
                          '("clustering.scrbl" "spectral.scrbl"
                                               "hdbscan.scrbl"
                                               "preprocessing.scrbl"
                                               "decomposition.scrbl"
                                               "manifold.scrbl"
                                               "estimators.scrbl"
                                               "pca-estimator.scrbl"
                                               "truncated-svd-estimator.scrbl"))))

(struct argument (keyword name default) #:transparent)

(define (arguments-of name)
  (for/list ([spec (in-list (dict-ref documented name))])
    (match spec
      [(list (? keyword? kw) id _contract default) (argument kw id default)]
      [(list (? keyword? kw) id _contract) (argument kw id 'none)]
      [(list id _contract default) (argument #f id default)]
      [(list id _contract) (argument #f id 'none)])))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(struct binding (name proc table keys taken keywords positional others))

(define (table-bindings table keys entries)
  (for/list ([entry (in-list entries)])
    (match-define (list* name proc extra positional others omitted leading) entry)
    (define taken (filter (lambda (key) (not (memq key omitted))) keys))
    (binding name
             proc
             table
             keys
             taken
             (append leading (map keyword-of taken) extra)
             positional
             others)))

(define kmeans-keys
  '(init n-clusters
         max-iter
         tol
         n-init
         random-state
         oversampling-factor))

(define dbscan-keys '(eps min-samples metric out-dtype max-mbytes-per-batch))

(define single-linkage-keys '(n-clusters metric linkage connectivity c))

(define spectral-clustering-keys
  '(n-clusters n-components random-state n-neighbors n-init eigen-tol affinity))

(define hdbscan-keys
  '(min-cluster-size min-samples
                     cluster-selection-epsilon
                     max-cluster-size
                     metric
                     alpha
                     cluster-selection-method
                     allow-single-cluster
                     build-algo))

(define standard-scaler-keys '(with-mean with-std))

(define pca-keys '(n-components whiten solver tol iterations))

(define tsvd-keys '(n-components solver tol iterations))

(define umap-keys
  '(n-neighbors n-components
                metric
                n-epochs
                learning-rate
                min-dist
                spread
                a
                b
                init
                negative-sample-rate
                random-state
                build-algo))

(define (estimator-entry name proc)
  (list name
        proc
        '(#:resources)
        (list (argument #f 'params #f))
        (list (argument '#:resources 'resources #f))
        '()))

(define (fit-entry name proc)
  (list name
        proc
        '(#:params #:resources)
        (list (argument #f 'X 'none))
        (list (argument '#:params 'params #f)
              (argument '#:resources 'resources '(current-device-resources)))
        '()))

(define (supervised-fit-entry name proc)
  (list name
        proc
        '(#:params #:resources)
        (list (argument #f 'X 'none))
        (list (argument '#:y 'y #f)
              (argument '#:params 'params #f)
              (argument '#:resources 'resources '(current-device-resources)))
        '()
        '#:y))

(define bindings
  (append
   (table-bindings kmeans-params
                   kmeans-keys
                   (list (list 'kmeans-params kmeans-params '() (list (argument #f 'base #f)) '() '())
                         (list 'kmeans-fit
                               kmeans-fit
                               '(#:params #:sample-weight #:max-samples-per-batch #:resources)
                               (list (argument #f 'X 'none))
                               (list (argument '#:params 'params #f)
                                     (argument '#:max-samples-per-batch
                                               'max-samples-per-batch
                                               '(current-batch-samples))
                                     (argument '#:resources 'resources '(current-device-resources)))
                               '())
                         (list 'kmeans
                               kmeans
                               '(#:max-samples-per-batch #:resources)
                               (list (argument #f 'params #f))
                               (list (argument '#:max-samples-per-batch 'max-samples-per-batch #f)
                                     (argument '#:resources 'resources #f))
                               '())))
   (table-bindings dbscan-params
                   dbscan-keys
                   (list (list 'dbscan-params dbscan-params '() (list (argument #f 'base #f)) '() '())
                         (list 'dbscan-fit
                               dbscan-fit
                               '(#:params #:sample-weight #:resources)
                               (list (argument #f 'X 'none))
                               (list (argument '#:params 'params #f)
                                     (argument '#:resources 'resources '(current-device-resources)))
                               '())
                         (estimator-entry 'dbscan dbscan)))
   (table-bindings
    single-linkage-params
    single-linkage-keys
    (list
     (list 'single-linkage-params single-linkage-params '() (list (argument #f 'base #f)) '() '())
     (list 'single-linkage-fit
           single-linkage-fit
           '(#:params #:resources)
           (list (argument #f 'X 'none))
           (list (argument '#:params 'params #f)
                 (argument '#:resources 'resources '(current-device-resources)))
           '(linkage))
     (estimator-entry 'agglomerative-clustering agglomerative-clustering)))
   (table-bindings spectral-clustering-params
                   spectral-clustering-keys
                   (list (list 'spectral-clustering-params
                               spectral-clustering-params
                               '()
                               (list (argument #f 'base #f))
                               '()
                               '())
                         (list 'spectral-clustering-fit
                               spectral-clustering-fit
                               '(#:params #:resources)
                               (list (argument #f 'X 'none))
                               (list (argument '#:params 'params #f)
                                     (argument '#:resources 'resources '(current-device-resources)))
                               '(affinity))
                         (list 'spectral-clustering-fit-graph
                               spectral-clustering-fit-graph
                               '(#:params #:resources)
                               (list (argument #f 'rows 'none)
                                     (argument #f 'cols 'none)
                                     (argument #f 'vals 'none)
                                     (argument #f 'n-samples 'none))
                               (list (argument '#:params 'params #f)
                                     (argument '#:resources 'resources '(current-device-resources)))
                               '(n-neighbors affinity))
                         (estimator-entry 'spectral-clustering spectral-clustering)))
   (table-bindings
    hdbscan-params
    hdbscan-keys
    (list (list 'hdbscan-params hdbscan-params '() (list (argument #f 'base #f)) '() '())
          (list 'hdbscan-fit
                hdbscan-fit
                '(#:params #:resources)
                (list (argument #f 'X 'none))
                (list (argument '#:params 'params #f)
                      (argument '#:resources 'resources '(current-device-resources)))
                '(metric build-algo))
          (estimator-entry 'hdbscan hdbscan)))
   (table-bindings
    standard-scaler-params
    standard-scaler-keys
    (list
     (list 'standard-scaler-params standard-scaler-params '() (list (argument #f 'base #f)) '() '())
     (fit-entry 'standard-scaler-fit standard-scaler-fit)
     (fit-entry 'standard-scaler-fit-transform standard-scaler-fit-transform)
     (estimator-entry 'standard-scaler standard-scaler)))
   (table-bindings pca-params
                   pca-keys
                   (list (list 'pca-params pca-params '() (list (argument #f 'base #f)) '() '())
                         (fit-entry 'pca-fit pca-fit)
                         (fit-entry 'pca-fit-transform pca-fit-transform)
                         (estimator-entry 'pca pca)))
   (table-bindings tsvd-params
                   tsvd-keys
                   (list (list 'tsvd-params tsvd-params '() (list (argument #f 'base #f)) '() '())
                         (fit-entry 'tsvd-fit tsvd-fit)
                         (fit-entry 'tsvd-fit-transform tsvd-fit-transform)
                         (estimator-entry 'truncated-svd truncated-svd)))
   (table-bindings umap-params
                   umap-keys
                   (list (list 'umap-params umap-params '() (list (argument #f 'base #f)) '() '())
                         (supervised-fit-entry 'umap-fit umap-fit)
                         (supervised-fit-entry 'umap-fit-transform umap-fit-transform)
                         (estimator-entry 'umap umap)))))

(define-namespace-anchor anchor)

(define (evaluated default bound)
  (parameterize ([current-namespace (make-base-empty-namespace)])
    (namespace-attach-module (namespace-anchor->empty-namespace anchor) 'cuml)
    (namespace-require 'racket/base)
    (namespace-require 'raft)
    (namespace-require 'cuml)
    (eval `(let* ,(for/list ([pair (in-list bound)])
                    (match-define (cons key value) pair)
                    `[,key (quote ,value)])
             ,default))))

(test-case "each documented table is its -params procedure's keys"
  (for ([b (in-list bindings)])
    (check-equal? (sort (binding-keys b) symbol<?)
                  (sort (hash-keys ((binding-table b))) symbol<?)
                  (format "~a" (binding-name b)))))

(test-case "each table-backed entry documents the table's keywords in order, then the others"
  (for ([b (in-list bindings)])
    (check-equal? (filter values (map argument-keyword (arguments-of (binding-name b))))
                  (binding-keywords b)
                  (format "~a" (binding-name b)))))

(test-case "each table-backed binding takes exactly the documented keywords"
  (for ([b (in-list bindings)])
    (define-values (required allowed) (procedure-keywords (binding-proc b)))
    (check-equal? required '() (format "~a" (binding-name b)))
    (check-equal? allowed (sort (binding-keywords b) keyword<?) (format "~a" (binding-name b)))))

(test-case "each documented hyperparameter default is the table's"
  (for ([b (in-list bindings)])
    (define name (binding-name b))
    (define defaults ((binding-table b)))
    (define by-keyword
      (for/hasheq ([a (in-list (arguments-of name))]
                   #:when (argument-keyword a))
        (values (argument-keyword a) a)))
    (for ([key (in-list (binding-taken b))])
      (define bound
        (for/list ([earlier (in-list (takef (binding-keys b) (lambda (k) (not (eq? k key)))))])
          (cons earlier (hash-ref defaults earlier))))
      (define entry (hash-ref by-keyword (keyword-of key)))
      (check-equal? (argument-name entry) key (format "~a #:~a" name key))
      (check-equal? (evaluated (argument-default entry) bound)
                    (hash-ref defaults key)
                    (format "~a #:~a" name key)))))

(test-case "the positional arguments and the defaults outside the table"
  (for ([b (in-list bindings)])
    (define name (binding-name b))
    (define documented-arguments (arguments-of name))
    (check-equal? (filter (lambda (a) (not (argument-keyword a))) documented-arguments)
                  (binding-positional b)
                  (format "~a" name))
    (for ([other (in-list (binding-others b))])
      (check-not-false (member other documented-arguments) (format "~a ~a" name other)))))
