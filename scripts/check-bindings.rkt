#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map count last remove-duplicates)
         (only-in racket/match == match match-lambda)
         (only-in racket/port with-output-to-string)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in racket/string string-prefix? string-split)
         (only-in racket/system system*)
         (only-in threading ~>))

(define-runtime-path root "..")

(define family "cluster")
(define prefix (format "cuml_~a_" family))
(define failure-kinds '(UNBOUND MISSING TWICE UNPLANNED DUE STALE-PLAN))

(struct planned (name status issue))

(define (unprefixed c-id)
  (and (string-prefix? c-id prefix) (substring c-id (string-length prefix))))

(define (nm-output library)
  (define nm (or (find-executable-path "nm") (error 'check-bindings "nm is not on PATH")))
  (with-output-to-string (lambda ()
                           (unless (system* nm "-D" "--defined-only" (path->string library))
                             (error 'check-bindings "nm failed on ~a" library)))))

(define (exports-in nm-text)
  (~> (for/list ([line (in-list (string-split nm-text "\n"))])
        (match (string-split line)
          [(list _ _ symbol) (unprefixed symbol)]
          [_ #f]))
      (filter values _)
      (sort string<?)))

(define (c-id-of name options)
  (match options
    [(list* '#:c-id id _) (symbol->string id)]
    [(list* _ rest) (c-id-of name rest)]
    ['() (regexp-replace* #rx"-" (symbol->string name) "_")]))

(define (form-bindings form)
  (match form
    [(list* 'define-job _ _ (? string? c-id) _) (list c-id)]
    [(list* 'define-cluster (? symbol? name) options) (list (c-id-of name options))]
    [(list* 'entry (? string? name) _) (list (string-append prefix name))]
    [(? list?) (append-map form-bindings form)]
    [_ '()]))

(define (module-bindings path)
  (define form
    (parameterize ([read-accept-reader #t]
                   [read-accept-lang #t])
      (call-with-input-file path read)))
  (filter-unprefixed (form-bindings form)))

(define (filter-unprefixed c-ids)
  (filter values (map unprefixed c-ids)))

(define (racket-module? path)
  (and (regexp-match? #rx"[.]rkt$" path) (not (regexp-match? #rx"/compiled/" path))))

(define (tree-bindings dir)
  (~> (find-files racket-module? dir) (sort path<?) (append-map module-bindings _)))

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
  (define bodies (job-bodies exports bindings))
  (define (with-status status present?)
    (for/list ([p (in-list entries)]
               #:when (eq? (planned-status p) status)
               #:when (eq? (and (member (planned-name p) exports) #t) present?))
      (planned-name p)))
  (define (findings kind names)
    (for/list ([n (in-list names)])
      (list kind n)))
  (append (findings 'UNBOUND
                    (for/list ([e (in-list exports)]
                               #:unless (member e bindings)
                               #:unless (member e bodies))
                      e))
          (findings 'MISSING
                    (for/list ([b (in-list (sort (remove-duplicates bindings) string<?))]
                               #:unless (member b exports))
                      b))
          (findings 'TWICE
                    (for/list ([b (in-list (sort (remove-duplicates bindings) string<?))]
                               #:when (< 1 (count (lambda (c) (equal? c b)) bindings)))
                      b))
          (findings 'UNPLANNED
                    (for/list ([e (in-list exports)]
                               #:unless (member e (map planned-name entries))
                               #:unless (member (string-append e "_job") (map planned-name entries)))
                      e))
          (findings 'DUE (with-status 'due #f))
          (findings 'STALE-PLAN (with-status 'planned #t))
          (findings 'PLANNED (with-status 'planned #f))))

(define (failed? findings)
  (for/or ([f (in-list findings)])
    (match f
      [(list kind _) (memq kind failure-kinds)])))

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

(module+ main
  (define (in-root . parts)
    (apply build-path root parts))
  (define library
    (in-root (format "cuml-~a-lib/cuml/~a/native-libs/libcumlrkt_~a.so" family family family)))
  (define plan (call-with-input-file (in-root (format "scripts/~a-bindings.rktd" family)) read))
  (define exports (exports-in (nm-output library)))
  (define bindings
    (append (tree-bindings (in-root (format "cuml-~a-lib/cuml/~a/private" family family)))
            (module-bindings (in-root "cuml-core-lib/cuml/core/private/native.rkt"))))
  (define findings (census exports bindings plan))
  (for ([f (in-list findings)])
    (match f
      [(list kind name) (printf "~a ~a~a\n" kind prefix name)]))
  (printf "~a: ~a exports, ~a bindings, ~a job bodies; ~a planned (~a)\n"
          family
          (length exports)
          (length bindings)
          (length (job-bodies exports bindings))
          (count-of 'PLANNED findings)
          (issue-range plan))
  (when (failed? findings)
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

  (test-case "bindings come from define-cluster, its #:c-id and the define-job string"
    (define dir (make-temporary-directory "check-bindings-~a"))
    (dynamic-wind
     void
     (lambda ()
       (make-directory* (build-path dir "private"))
       (with-output-to-file
        (build-path dir "private" "native.rkt")
        (lambda ()
          (write '(module native racket/base
                    (define-cluster cuml-cluster-make-blobs (_fun -> _int))
                    (define-cluster blobs-short (_fun -> _int) #:c-id cuml_cluster_short_blobs)
                    (define-job cuml-cluster-fit-job library "cuml_cluster_kmeans_fit_job" (handle))
                    (define-job other-job library "raft_elsewhere" (handle))))))
       (check-equal? (tree-bindings dir) '("make_blobs" "short_blobs" "kmeans_fit_job")))
     (lambda () (delete-directory/files dir #:must-exist? #f))))

  (test-case "nm lines become unprefixed export names"
    (check-equal?
     (exports-in "0000 T cuml_cluster_make_blobs\n0000 T rr_other\n         U cuml_cluster_x\n")
     '("make_blobs")))

  (test-case "the summary names the planned issues"
    (check-equal? (issue-range plan) "issue 8")
    (check-equal? (issue-range (append plan '((issue 12 planned hdbscan_fit_job)))) "issues 8-12")))
