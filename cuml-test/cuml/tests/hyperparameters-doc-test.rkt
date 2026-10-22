#lang racket/base

(require (only-in cuml kmeans)
         (only-in cuml/cluster kmeans-fit kmeans-params)
         (only-in racket/list append-map)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in scribble/reader read-inside))

(define (manual-forms file)
  (call-with-input-file (collection-file-path file "cuml" "scribblings")
                        (lambda (in)
                          (read-line in 'any)
                          (read-inside in))))

(define (defprocs datum)
  (match datum
    [(cons 'defproc (cons (cons (? symbol? name) arguments) _)) (list (cons name arguments))]
    [(? pair?) (append (defprocs (car datum)) (defprocs (cdr datum)))]
    [_ '()]))

(define documented
  (append-map defprocs (append (manual-forms "clustering.scrbl") (manual-forms "estimators.scrbl"))))

(struct argument (keyword name default) #:transparent)

(define (arguments-of name)
  (for/list ([spec (in-list (cdr (assq name documented)))])
    (match spec
      [(list (? keyword? kw) id _contract default) (argument kw id default)]
      [(list (? keyword? kw) id _contract) (argument kw id 'none)]
      [(list id _contract default) (argument #f id default)]
      [(list id _contract) (argument #f id 'none)])))

(define (keyword-of key)
  (string->keyword (symbol->string key)))

(define table-keys
  '(init n-clusters
         max-iter
         tol
         n-init
         random-state
         oversampling-factor))

(define table-keywords (map keyword-of table-keys))

(define bindings
  (list
   (list 'kmeans-params kmeans-params table-keywords (list (argument #f 'base #f)) '())
   (list 'kmeans-fit
         kmeans-fit
         (append table-keywords '(#:params #:sample-weight #:max-samples-per-batch #:resources))
         (list (argument #f 'X 'none))
         (list (argument '#:params 'params #f)
               (argument '#:max-samples-per-batch 'max-samples-per-batch '(current-batch-samples))
               (argument '#:resources 'resources '(current-device-resources))))
   (list 'kmeans
         kmeans
         (append table-keywords '(#:max-samples-per-batch #:resources))
         (list (argument #f 'params #f))
         (list (argument '#:max-samples-per-batch 'max-samples-per-batch #f)
               (argument '#:resources 'resources #f)))))

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

(test-case "the documented table is kmeans-params' keys"
  (check-equal? (sort table-keys symbol<?) (sort (hash-keys (kmeans-params)) symbol<?)))

(test-case "each table-backed entry documents the table's keywords in order, then the others"
  (for ([binding (in-list bindings)])
    (match-define (list name _ keywords _ _) binding)
    (check-equal? (filter values (map argument-keyword (arguments-of name)))
                  keywords
                  (format "~a" name))))

(test-case "each table-backed binding takes exactly the documented keywords"
  (for ([binding (in-list bindings)])
    (match-define (list name proc keywords _ _) binding)
    (define-values (required allowed) (procedure-keywords proc))
    (check-equal? required '() (format "~a" name))
    (check-equal? allowed (sort keywords keyword<?) (format "~a" name))))

(test-case "each documented hyperparameter default is the table's"
  (for ([binding (in-list bindings)])
    (match-define (list name _ _ _ _) binding)
    (define by-keyword
      (for/hasheq ([a (in-list (arguments-of name))]
                   #:when (argument-keyword a))
        (values (argument-keyword a) a)))
    (for/fold ([bound '()]) ([key (in-list table-keys)])
      (define value (hash-ref (kmeans-params) key))
      (define entry (hash-ref by-keyword (keyword-of key)))
      (check-equal? (argument-name entry) key (format "~a #:~a" name key))
      (check-equal? (evaluated (argument-default entry) bound) value (format "~a #:~a" name key))
      (append bound (list (cons key value))))))

(test-case "the positional arguments and the defaults outside the table"
  (for ([binding (in-list bindings)])
    (match-define (list name _ _ positional others) binding)
    (define documented-arguments (arguments-of name))
    (check-equal? (filter (lambda (a) (not (argument-keyword a))) documented-arguments)
                  positional
                  (format "~a" name))
    (for ([other (in-list others)])
      (check-true (and (member other documented-arguments) #t) (format "~a ~a" name other)))))
