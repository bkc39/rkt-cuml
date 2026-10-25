#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/function negate)
         (only-in racket/list append-map count filter-map last remove-duplicates)
         (only-in racket/match == match match-define match-lambda)
         (only-in racket/port with-output-to-string)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in racket/string string-prefix? string-split)
         (only-in racket/system system*)
         (only-in threading ~>))

(define-runtime-path root "..")

(define families '("cluster" "metrics" "preprocessing"))
(define failure-kinds '(UNBOUND MISSING TWICE UNPLANNED DUE STALE-PLAN))

(struct planned (name status issue))

(define (prefix-of family)
  (format "cuml_~a_" family))

(define ((unprefixed family) c-id)
  (define prefix (prefix-of family))
  (and (string-prefix? c-id prefix) (substring c-id (string-length prefix))))

(define (nm-output library)
  (define nm (or (find-executable-path "nm") (error 'check-bindings "nm is not on PATH")))
  (with-output-to-string (lambda ()
                           (unless (system* nm "-D" "--defined-only" (path->string library))
                             (error 'check-bindings "nm failed on ~a" library)))))

(define (exports-in family nm-text)
  (~> (filter-map (lambda (line)
                    (match (string-split line)
                      [(list _ _ symbol) ((unprefixed family) symbol)]
                      [_ #f]))
                  (string-split nm-text "\n"))
      (sort string<?)))

(define (c-id-of name options)
  (match options
    [(list* '#:c-id id _) (symbol->string id)]
    [(list* _ rest) (c-id-of name rest)]
    ['() (regexp-replace* #rx"-" (symbol->string name) "_")]))

(define ((form-bindings family) form)
  (define definer (string->symbol (format "define-~a" family)))
  (match form
    [(list* 'define-job _ _ (? string? c-id) _) (list c-id)]
    [(list* (== definer) (? symbol? name) options) (list (c-id-of name options))]
    [(list* 'entry (? string? name) _) (list (string-append (prefix-of family) name))]
    [(? list?) (append-map (form-bindings family) form)]
    [_ '()]))

(define (module-bindings family path)
  (define form
    (parameterize ([read-accept-reader #t]
                   [read-accept-lang #t])
      (call-with-input-file path read)))
  (filter-map (unprefixed family) ((form-bindings family) form)))

(define (racket-module? path)
  (and (regexp-match? #rx"[.]rkt$" path) (not (regexp-match? #rx"/compiled/" path))))

(define (tree-bindings family dir)
  (~> (find-files racket-module? dir)
      (sort path<?)
      (append-map (lambda (path) (module-bindings family path)) _)))

(define (plan-entries plan)
  (define (entries names status issue)
    (for/list ([n (in-list names)])
      (planned (symbol->string n) status issue)))
  (append-map (match-lambda
                [(list* 'common names) (entries names 'due 'common)]
                [(list* 'scaffold status names) (entries names status 'scaffold)]
                [(list* 'issue issue status names) (entries names status issue)])
              plan))

(define (job-bodies exports bindings)
  (for/list ([e (in-list exports)]
             #:when (member (string-append e "_job") exports)
             #:when (member (string-append e "_job") bindings)
             #:unless (member e bindings))
    e))

(define (census exports bindings plan)
  (define entries (plan-entries plan))
  (define names (map planned-name entries))
  (define bodies (job-bodies exports bindings))
  (define distinct (sort (remove-duplicates bindings) string<?))
  (define (exported? p)
    (member (planned-name p) exports))
  (define (named status keep?)
    (for/list ([p (in-list entries)]
               #:when (eq? (planned-status p) status)
               #:when (keep? p))
      (planned-name p)))
  (define (findings kind found)
    (for/list ([n (in-list found)])
      (list kind n)))
  (append (findings 'UNBOUND
                    (for/list ([e (in-list exports)]
                               #:unless (member e bindings)
                               #:unless (member e bodies))
                      e))
          (findings 'MISSING
                    (for/list ([b (in-list distinct)]
                               #:unless (member b exports))
                      b))
          (findings 'TWICE
                    (for/list ([b (in-list distinct)]
                               #:when (< 1 (count (lambda (c) (equal? c b)) bindings)))
                      b))
          (findings 'UNPLANNED
                    (for/list ([e (in-list exports)]
                               #:unless (member e names)
                               #:unless (member (string-append e "_job") names))
                      e))
          (findings 'DUE (named 'due (negate exported?)))
          (findings 'STALE-PLAN (named 'planned exported?))
          (findings 'PLANNED (named 'planned (negate exported?)))))

(define (failed? findings)
  (ormap (match-lambda
           [(list kind _) (memq kind failure-kinds)])
         findings))

(define (issue-range plan)
  (define issues
    (~> (for/list ([p (in-list (plan-entries plan))]
                   #:when (eq? (planned-status p) 'planned))
          (planned-issue p))
        remove-duplicates
        (sort <)))
  (match issues
    ['() "none"]
    [(list one) (format "issue ~a" one)]
    [(cons first more) (format "issues ~a-~a" first (last more))]))

(define (count-of kind findings)
  (for/sum ([f (in-list findings)])
           (match f
             [(list (== kind) _) 1]
             [_ 0])))

(define (family-census in-root family)
  (define library
    (in-root (format "cuml-~a-lib/cuml/~a/native-libs/libcumlrkt_~a.so" family family family)))
  (define plan (call-with-input-file (in-root (format "scripts/~a-bindings.rktd" family)) read))
  (define exports (exports-in family (nm-output library)))
  (define bindings
    (append (tree-bindings family (in-root (format "cuml-~a-lib/cuml/~a/private" family family)))
            (module-bindings family (in-root "cuml-core-lib/cuml/core/private/native.rkt"))))
  (define findings (census exports bindings plan))
  (for ([f (in-list findings)])
    (match-define (list kind name) f)
    (printf "~a ~a~a\n" kind (prefix-of family) name))
  (printf "~a: ~a exports, ~a bindings, ~a job bodies; ~a planned (~a)\n"
          family
          (length exports)
          (length bindings)
          (length (job-bodies exports bindings))
          (count-of 'PLANNED findings)
          (issue-range plan))
  (failed? findings))

(module+ main
  (define (in-root . parts)
    (apply build-path root parts))
  (define failures
    (for/list ([family (in-list families)]
               #:when (family-census in-root family))
      family))
  (unless (null? failures)
    (exit 1)))

(module+ test
  (require (only-in racket/file delete-directory/files make-directory* make-temporary-directory)
           (only-in rackunit check-equal? check-false check-true test-case))

  (define plan
    '((common last_error) (scaffold due make_blobs make_blobs_job)
                          (issue 6 due kmeans_fit_job)
                          (issue 8 planned dbscan_fit_job)))

  (define clean-exports '("kmeans_fit" "kmeans_fit_job" "last_error" "make_blobs" "make_blobs_job"))
  (define clean-bindings '("make_blobs" "make_blobs_job" "kmeans_fit_job" "last_error"))

  (define (kinds findings)
    (filter (match-lambda
              [(list kind _) (not (eq? kind 'PLANNED))])
            findings))

  (test-case "due entries that are exported and bound pass, and a planned absent one is only reported"
    (define findings (census clean-exports clean-bindings plan))
    (check-equal? (kinds findings) '())
    (check-equal? findings '((PLANNED "dbscan_fit_job")))
    (check-false (failed? findings)))

  (test-case "a body beside its bound job passes; without the job it is UNBOUND"
    (check-equal? (job-bodies clean-exports clean-bindings) '("kmeans_fit"))
    (check-equal? (kinds (census clean-exports (remove "kmeans_fit_job" clean-bindings) plan))
                  '((UNBOUND "kmeans_fit") (UNBOUND "kmeans_fit_job"))))

  (test-case "each failure kind fires"
    (define findings
      (census (append clean-exports '("dbscan_fit_job" "stray"))
              (append clean-bindings '("make_blobs" "gone"))
              (append plan '((issue 9 due single_linkage_job)))))
    (check-equal? (kinds findings)
                  '((UNBOUND "dbscan_fit_job") (UNBOUND "stray")
                                               (MISSING "gone")
                                               (TWICE "make_blobs")
                                               (UNPLANNED "stray")
                                               (DUE "single_linkage_job")
                                               (STALE-PLAN "dbscan_fit_job")))
    (check-true (and (failed? findings) #t)))

  (define (call-with-temporary-directory proc)
    (define dir (make-temporary-directory "check-bindings-~a"))
    (dynamic-wind void
                  (lambda () (proc dir))
                  (lambda () (delete-directory/files dir #:must-exist? #f))))

  (test-case "bindings come from define-<family>, its #:c-id and the define-job string"
    (call-with-temporary-directory
     (lambda (dir)
       (make-directory* (build-path dir "private"))
       (with-output-to-file
        (build-path dir "private" "native.rkt")
        (lambda ()
          (write '(module native racket/base
                    (define-cluster cuml-cluster-make-blobs (_fun -> _int))
                    (define-cluster blobs-short (_fun -> _int) #:c-id cuml_cluster_short_blobs)
                    (define-job cuml-cluster-fit-job library "cuml_cluster_kmeans_fit_job" (handle))
                    (define-job other-job library "raft_elsewhere" (handle))))))
       (check-equal? (tree-bindings "cluster" dir) '("make_blobs" "short_blobs" "kmeans_fit_job")))))

  (test-case "each family counts only its own prefix and its own definer"
    (call-with-temporary-directory
     (lambda (dir)
       (make-directory* (build-path dir "private"))
       (with-output-to-file
        (build-path dir "private" "native.rkt")
        (lambda ()
          (write '(module native racket/base
                    (define-cluster cuml-cluster-make-blobs (_fun -> _int))
                    (define-metrics cuml-metrics-version (_fun -> _int))
                    (define-cluster cuml-metrics-stray (_fun -> _int))
                    (define-job ari-job library "cuml_metrics_adjusted_rand_score_job" (handle))
                    (define-job fit-job library "cuml_cluster_kmeans_fit_job" (handle))
                    (entry "last_error" (_fun -> _int))))))
       (check-equal? (tree-bindings "cluster" dir) '("make_blobs" "kmeans_fit_job" "last_error"))
       (check-equal? (tree-bindings "metrics" dir)
                     '("version" "adjusted_rand_score_job" "last_error")))))

  (test-case "the family list holds cluster, metrics and preprocessing"
    (check-equal? families '("cluster" "metrics" "preprocessing")))

  (test-case "nm lines become unprefixed export names"
    (check-equal?
     (exports-in "cluster"
                 "0000 T cuml_cluster_make_blobs\n0000 T rr_other\n         U cuml_cluster_x\n")
     '("make_blobs"))
    (check-equal? (exports-in "metrics"
                              "0000 T cuml_cluster_make_blobs\n0000 T cuml_metrics_check_abi\n")
                  '("check_abi")))

  (test-case "the summary names the planned issues"
    (check-equal? (issue-range plan) "issue 8")
    (check-equal? (issue-range (append plan '((issue 12 planned hdbscan_fit_job)))) "issues 8-12")))
