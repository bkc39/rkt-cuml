#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/function curry)
         (only-in racket/list append-map filter-map)
         (only-in racket/match match)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in setup/getinfo get-info/full)
         (only-in threading ~>))

(provide families)

(define-runtime-path root "..")

(define family-allowed-deps '("base" "raft" "cuml-core-lib"))
(define core-allowed-deps '("base" "raft"))
(define umbrella-base-deps '("base" "raft" "cuml-core-lib" "datasets-core"))
(define dependency-fields '(deps build-deps implies))
(define reserved-names
  '(agglomerative-clustering dbscan hdbscan kmeans single-linkage spectral-clustering))

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define (package-deps dir package)
  (define info (get-info/full (build-path dir package)))
  (for*/list ([field (in-list dependency-fields)]
              [dep (in-list (info field (lambda () '())))])
    (dep-name dep)))

(define (family-package family)
  (format "cuml-~a-lib" family))

(define (family-of package)
  (match (regexp-match #rx"^cuml-(.+)-lib$" package)
    [(list _ (and family (not "core"))) family]
    [_ #f]))

(define (families dir)
  (~> (directory-list dir) (map path->string _) (filter-map family-of _) (sort string<?)))

(define (deps-problems package deps allowed)
  (for/list ([dep (in-list deps)]
             #:unless (member dep allowed))
    (format "~a depends on ~a; it may depend only on ~a" package dep allowed)))

(define (missing-problems package deps wanted)
  (for/list ([want (in-list wanted)]
             #:unless (member want deps))
    (format "~a does not depend on ~a" package want)))

(define (reserved-problems module names)
  (for/list ([name (in-list names)]
             #:when (memq name reserved-names))
    (format "~a exports ~a, a name reserved for cuml's estimator wrappers" module name)))

(define (public-module-file? path)
  (and (regexp-match? #rx"[.]rkt$" path)
       (not (regexp-match? #rx"/(private|compiled)/" path))
       (not (regexp-match? #rx"/info[.]rkt$" path))))

(define (public-modules dir family)
  (define collection (build-path dir (family-package family) "cuml"))
  (define nested (build-path collection family))
  (define nested-modules
    (if (directory-exists? nested)
        (find-files public-module-file? nested)
        '()))
  (cons (build-path collection (string-append family ".rkt")) (sort nested-modules path<?)))

(define (exported-names module)
  (dynamic-require module (void))
  (define-values (variables syntaxes) (module->exports module))
  (for*/list ([phase+names (in-list (append variables syntaxes))]
              #:when (eqv? (car phase+names) 0)
              [entry (in-list (cdr phase+names))])
    (car entry)))

(define (family-deps-problems dir family)
  (define package (family-package family))
  (deps-problems package (package-deps dir package) family-allowed-deps))

(define (core-public-modules dir)
  (define collection (build-path dir "cuml-core-lib" "cuml"))
  (if (directory-exists? collection)
      (sort (find-files public-module-file? collection) path<?)
      '()))

(define (exports-problems modules)
  (for*/list ([module (in-list modules)]
              [problem (in-list (reserved-problems (path->string module) (exported-names module)))])
    problem))

(define (family-exports-problems dir family)
  (exports-problems (public-modules dir family)))

(define (package-version dir)
  ((get-info/full dir) 'version (lambda () #f)))

(define (version-problems . dirs)
  (match dirs
    ['() '()]
    [(list dir-a dir-b) (versions-differ dir-a dir-b)]))

(define (versions-differ dir-a dir-b)
  (define version-a (package-version dir-a))
  (define version-b (package-version dir-b))
  (cond
    [(equal? version-a version-b) '()]
    [else (list (format "~a is version ~a but ~a is version ~a" dir-a version-a dir-b version-b))]))

(define (layout-problems dir #:exports? exports?)
  (define family-names (families dir))
  (append (append-map (curry family-deps-problems dir) family-names)
          (deps-problems "cuml-core-lib" (package-deps dir "cuml-core-lib") core-allowed-deps)
          (deps-problems "cuml-lib"
                         (package-deps dir "cuml-lib")
                         (append umbrella-base-deps (map family-package family-names)))
          (missing-problems "cuml-lib"
                            (package-deps dir "cuml-lib")
                            (cons "cuml-core-lib" (map family-package family-names)))
          (if exports?
              (append (exports-problems (core-public-modules dir))
                      (append-map (curry family-exports-problems dir) family-names))
              '())))

(module+ main
  (require (only-in racket/cmdline command-line))

  (define exports? #t)
  (define same-version '())
  (command-line #:once-each
                [("--deps-only") "check info.rkt deps only, loading no module" (set! exports? #f)]
                [("--same-version")
                 package-a
                 package-b
                 "check that two package directories have one version"
                 (set! same-version (list package-a package-b))])
  (define problems
    (append (layout-problems (simplify-path root) #:exports? exports?)
            (apply version-problems same-version)))
  (for-each displayln problems)
  (cond
    [(pair? problems) (exit 1)]
    [else
     (printf "check-layout: ~a family packages; deps~a clean\n"
             (length (families root))
             (if exports? " and exports" ""))]))

(module+ test
  (require (only-in racket/file delete-directory/files make-directory* make-temporary-directory)
           (only-in rackunit check-equal? test-case))

  (define (write-tree-file! dir file)
    (match file
      [(list path form)
       (define full (build-path dir path))
       (make-directory* (simplify-path (build-path full 'up)))
       (with-output-to-file full (lambda () (write form)))]))

  (define (call-with-tree files proc)
    (define dir (make-temporary-directory "check-layout-~a"))
    (dynamic-wind void
                  (lambda ()
                    (for-each (curry write-tree-file! dir) files)
                    (proc dir))
                  (lambda () (delete-directory/files dir #:must-exist? #f))))

  (define (info-file package field deps)
    (list (format "~a/info.rkt" package)
          `(module info info
             (define ,field ',deps))))

  (define (toy-tree field
                    toy-deps
                    #:core [core-deps '("base" "raft")]
                    #:umbrella [umbrella-deps '("base" "cuml-core-lib" "cuml-toy-lib")])
    (list (info-file "cuml-core-lib" 'deps core-deps)
          (info-file "cuml-lib" 'deps umbrella-deps)
          (info-file "cuml-toy-lib" field toy-deps)
          (list "cuml-toy-lib/cuml/toy.rkt"
                '(module toy racket/base
                   (provide kmeans
                            kmeans-fit)
                   (define kmeans 0)
                   (define kmeans-fit 0)))))

  (for ([field (in-list '(deps build-deps implies))])
    (test-case (format "a family's ~a may name only base, raft and cuml-core-lib" field)
      (check-equal?
       (call-with-tree (toy-tree field '("base" ("raft" #:version "0.2") "cuml-lib"))
                       (lambda (dir) (layout-problems dir #:exports? #f)))
       '("cuml-toy-lib depends on cuml-lib; it may depend only on (base raft cuml-core-lib)"))))

  (test-case "cuml-lib may depend on datasets-core, and on nothing else outside the repo"
    (check-equal?
     (call-with-tree (toy-tree 'deps
                               '("base" "cuml-core-lib")
                               #:umbrella '("base" ("raft" #:version "0.2")
                                                   "cuml-core-lib"
                                                   "cuml-toy-lib"
                                                   ("datasets-core" #:version "0.1")
                                                   "datasets"))
                     (lambda (dir) (layout-problems dir #:exports? #f)))
     '("cuml-lib depends on datasets; it may depend only on (base raft cuml-core-lib datasets-core cuml-toy-lib)")))

  (test-case "a family that depends on datasets-core is reported"
    (check-equal?
     (call-with-tree (toy-tree 'deps '("base" "cuml-core-lib" ("datasets-core" #:version "0.1")))
                     (lambda (dir) (layout-problems dir #:exports? #f)))
     '("cuml-toy-lib depends on datasets-core; it may depend only on (base raft cuml-core-lib)")))

  (test-case "core that depends on datasets-core is reported"
    (check-equal? (call-with-tree
                   (toy-tree 'deps '("base" "cuml-core-lib") #:core '("base" "raft" "datasets-core"))
                   (lambda (dir) (layout-problems dir #:exports? #f)))
                  '("cuml-core-lib depends on datasets-core; it may depend only on (base raft)")))

  (test-case "cuml-lib must depend on every family"
    (check-equal? (missing-problems "cuml-lib" '("base" "cuml-core-lib") '("cuml-cluster-lib"))
                  '("cuml-lib does not depend on cuml-cluster-lib")))

  (test-case "family packages are the cuml-*-lib directories but core"
    (check-equal?
     (filter-map family-of '("cuml-core-lib" "cuml-cluster-lib" "cuml-metrics-lib" "cuml-lib" "lint"))
     '("cluster" "metrics")))

  (test-case "a family module that provides a bare algorithm name is reported, and only that name"
    (call-with-tree
     (toy-tree 'deps '("base" "cuml-core-lib"))
     (lambda (dir)
       (check-equal? (layout-problems dir #:exports? #t)
                     (list (format "~a exports kmeans, a name reserved for cuml's estimator wrappers"
                                   (build-path dir "cuml-toy-lib/cuml/toy.rkt")))))))

  (test-case "a core module that provides a bare algorithm name is reported"
    (call-with-tree
     (append (toy-tree 'deps '("base" "cuml-core-lib"))
             (list (list "cuml-core-lib/cuml/core.rkt"
                         '(module core racket/base
                            (provide fit
                                     kmeans)
                            (define fit 0)
                            (define kmeans 0)))
                   (list "cuml-core-lib/cuml/core/private/native.rkt"
                         '(module native racket/base
                            (provide kmeans)
                            (define kmeans 0)))))
     (lambda (dir)
       (check-equal? (layout-problems dir #:exports? #t)
                     (list (format "~a exports kmeans, a name reserved for cuml's estimator wrappers"
                                   (build-path dir "cuml-core-lib/cuml/core.rkt"))
                           (format "~a exports kmeans, a name reserved for cuml's estimator wrappers"
                                   (build-path dir "cuml-toy-lib/cuml/toy.rkt")))))))

  (test-case "two package directories with different versions are reported"
    (call-with-tree (list (info-file "pin-a" 'version "2.0")
                          (info-file "pin-b" 'version "2.1")
                          (info-file "pin-c" 'version "2.0"))
                    (lambda (dir)
                      (define (in-tree package)
                        (build-path dir package))
                      (check-equal? (version-problems (in-tree "pin-a") (in-tree "pin-c")) '())
                      (check-equal? (version-problems (in-tree "pin-a") (in-tree "pin-b"))
                                    (list (format "~a is version 2.0 but ~a is version 2.1"
                                                  (in-tree "pin-a")
                                                  (in-tree "pin-b"))))))))
