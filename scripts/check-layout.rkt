#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map filter-map)
         (only-in racket/match match)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in setup/getinfo get-info/full))

(define-runtime-path root "..")

(define family-allowed-deps '("base" "raft" "cuml-core-lib"))
(define core-allowed-deps '("base" "raft"))
(define reserved-names
  '(agglomerative-clustering dbscan hdbscan kmeans single-linkage spectral-clustering))

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define (package-deps dir)
  (map dep-name ((get-info/full dir) 'deps (lambda () '()))))

(define (family-of package)
  (match (regexp-match #rx"^cuml-(.+)-lib$" package)
    [(list _ (and family (not "core"))) family]
    [_ #f]))

(define (families dir)
  (sort (filter-map family-of (map path->string (directory-list dir))) string<?))

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

(define (public-modules dir family)
  (define collection (build-path dir (format "cuml-~a-lib" family) "cuml"))
  (define nested (build-path collection family))
  (cons (build-path collection (string-append family ".rkt"))
        (if (directory-exists? nested)
            (sort (find-files (lambda (p)
                                (and (regexp-match? #rx"[.]rkt$" p)
                                     (not (regexp-match? #rx"/(private|compiled)/" p))
                                     (not (regexp-match? #rx"/info[.]rkt$" p))))
                              nested)
                  path<?)
            '())))

(define (exported-names module)
  (void (module-declared? module #t))
  (define-values (variables syntaxes) (module->exports module))
  (for*/list ([phase+names (in-list (append variables syntaxes))]
              #:when (eqv? (car phase+names) 0)
              [entry (in-list (cdr phase+names))])
    (car entry)))

(define (layout-problems dir #:exports? exports?)
  (define family-names (families dir))
  (define (package-dir name)
    (build-path dir name))
  (append
   (append-map (lambda (family)
                 (define package (format "cuml-~a-lib" family))
                 (deps-problems package (package-deps (package-dir package)) family-allowed-deps))
               family-names)
   (deps-problems "cuml-core-lib" (package-deps (package-dir "cuml-core-lib")) core-allowed-deps)
   (missing-problems "cuml-lib"
                     (package-deps (package-dir "cuml-lib"))
                     (cons "cuml-core-lib" (map (lambda (f) (format "cuml-~a-lib" f)) family-names)))
   (if exports?
       (for*/list ([family (in-list family-names)]
                   [module (in-list (public-modules dir family))]
                   [problem (in-list (reserved-problems (path->string module)
                                                        (exported-names module)))])
         problem)
       '())))

(module+ main
  (require (only-in racket/cmdline command-line))

  (define exports? #t)
  (command-line #:once-each
                [("--deps-only") "check info.rkt deps only, loading no module" (set! exports? #f)])
  (define problems (layout-problems (simplify-path root) #:exports? exports?))
  (for-each displayln problems)
  (cond
    [(pair? problems) (exit 1)]
    [else
     (printf "check-layout: ~a family packages; deps~a clean\n"
             (length (families root))
             (if exports? " and exports" ""))]))

(module+ test
  (require (only-in rackunit check-equal? test-case))

  (test-case "a family package may depend only on base, raft and cuml-core-lib"
    (check-equal? (deps-problems "cuml-cluster-lib"
                                 (map dep-name '("base" ("raft" #:version "0.2") "cuml-core-lib"))
                                 family-allowed-deps)
                  '())
    (check-equal?
     (deps-problems "cuml-cluster-lib" '("base" "cuml-lib" "cuml-linear-lib") family-allowed-deps)
     (list
      "cuml-cluster-lib depends on cuml-lib; it may depend only on (base raft cuml-core-lib)"
      "cuml-cluster-lib depends on cuml-linear-lib; it may depend only on (base raft cuml-core-lib)")))

  (test-case "cuml-lib must depend on every family"
    (check-equal? (missing-problems "cuml-lib" '("base" "cuml-core-lib") '("cuml-cluster-lib"))
                  '("cuml-lib does not depend on cuml-cluster-lib")))

  (test-case "family packages are the cuml-*-lib directories but core"
    (check-equal? (filter-map family-of '("cuml-core-lib" "cuml-cluster-lib" "cuml-lib" "lint"))
                  '("cluster")))

  (test-case "a bare algorithm name is refused in a family's exports"
    (check-equal? (reserved-problems "cuml/cluster" '(kmeans-fit kmeans dbscan-fit))
                  '("cuml/cluster exports kmeans, a name reserved for cuml's estimator wrappers"))))
