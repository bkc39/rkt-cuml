#lang racket/base

(require (only-in cuml/scribblings/guide/source page-helpers-path page-helpers-source)
         (only-in racket/list append* append-map remove-duplicates)
         (only-in racket/match == match)
         (only-in racket/string string-replace)
         (only-in syntax/parse/pre ~or* attribute id kernel-literals syntax-parse)
         (only-in threading ~>))

(define helpers-page "examples.html")

(define linked-name #rx"<a [^>]*class=\"Rkt[A-Za-z]*Link\"[^>]*>([^<]*)</a>")

(define unlinked-name #rx"<span class=\"RktSym\">([^<]*)</span>")

(define field-name #rx"<span class=\"RktSym\">([^<]*)</span><span class=\"hspace\">&nbsp;</span>:")

(define (exported-names module-path)
  (module-declared? module-path #t)
  (define-values (variables syntaxes) (module->exports module-path))
  (for*/list ([exports (in-list (list variables syntaxes))]
              [name+origins (in-list (match (assv 0 exports)
                                       [(cons _ names) names]
                                       [#f '()]))])
    (symbol->string (car name+origins))))

(define (required-modules)
  (define-values (shown-require _helpers) (page-helpers-source))
  (match (syntax->datum shown-require)
    [(cons 'require module-paths) module-paths]))

(define (example-names modules)
  (for*/hash ([module-path (in-list (cons 'racket/base modules))]
              [name (in-list (exported-names module-path))]
              #:unless (member name (map symbol->string modules)))
    (values name #t)))

(define (formal-identifiers formals)
  (syntax-parse formals
    [(name:id ...) (attribute name)]
    [(name:id ... . rest:id) (cons #'rest (attribute name))]))

(define (clause-identifiers clause)
  (syntax-parse clause
    [(formals body ...)
     (append (formal-identifiers #'formals) (append-map bound-identifiers (attribute body)))]))

(define (bound-identifiers form)
  (syntax-parse form
    #:literal-sets (kernel-literals)
    [((~or* quote quote-syntax) . _) '()]
    [(define-values (name ...) rhs) (append (attribute name) (bound-identifiers #'rhs))]
    [(#%plain-lambda . clause) (clause-identifiers #'clause)]
    [(case-lambda
       clause ...)
     (append-map clause-identifiers (attribute clause))]
    [((~or* let-values letrec-values) ([(name ...) rhs] ...) body ...)
     (append (append* (attribute name))
             (append-map bound-identifiers (attribute rhs))
             (append-map bound-identifiers (attribute body)))]
    [(part ...) (append-map bound-identifiers (attribute part))]
    [_ '()]))

(define (written-in-helpers? identifier)
  (and (syntax-original? identifier) (equal? (syntax-source identifier) page-helpers-path)))

(define (helper-bindings)
  (define-values (shown-require helpers) (page-helpers-source))
  (define page-module (datum->syntax #f (list* 'module 'page 'racket/base shown-require helpers)))
  (~> (parameterize ([current-namespace (make-base-namespace)])
        (expand page-module))
      bound-identifiers
      (filter written-in-helpers? _)
      (map (lambda (identifier) (symbol->string (syntax-e identifier))) _)
      remove-duplicates))

(define (unescaped html)
  (for/fold ([text html]) ([entity (in-list '(("&lt;" . "<") ("&gt;" . ">") ("&amp;" . "&")))])
    (string-replace text (car entity) (cdr entity))))

(define (rendered-names pattern html)
  (remove-duplicates (map unescaped (regexp-match* pattern html #:match-select cadr))))

(define (html-findings labelled bindings modules page html)
  (define shown-bindings
    (match page
      [(== helpers-page) bindings]
      [_ '()]))
  (define fields (rendered-names field-name html))
  (append (for/list ([name (in-list (rendered-names linked-name html))]
                     #:when (member name shown-bindings))
            (format "~a: ~a, which a helper binds, links" page name))
          (for/list ([name (in-list (rendered-names linked-name html))]
                     #:when (member name modules))
            (format "~a: ~a, a module the examples require, links to a binding" page name))
          (for/list ([name (in-list (rendered-names unlinked-name html))]
                     #:when (hash-ref labelled name #f)
                     #:unless (member name fields))
            (format "~a: ~a, which the examples can use, has no link" page name))))

(module+ main
  (require (only-in racket/file file->string))

  (define dir
    (match (current-command-line-arguments)
      [(vector dir) dir]
      [_ (raise-user-error 'check-doc-links "usage: racket scripts/check-doc-links.rkt <html-dir>")]))
  (define modules (required-modules))
  (define labelled (example-names modules))
  (define bindings (helper-bindings))
  (define findings
    (for*/list ([page (in-list (map path->string (directory-list dir)))]
                #:when (regexp-match? #rx"[.]html$" page)
                [finding
                 (in-list
                  (~> (build-path dir page)
                      file->string
                      (html-findings labelled bindings (map symbol->string modules) page _)))])
      finding))
  (for-each displayln findings)
  (match findings
    ['() (printf "check-doc-links: every name the examples can use links in ~a\n" dir)]
    [_ (exit 1)]))

(module+ test
  (require (only-in rackunit check-equal? check-false check-not-false test-case))

  (test-case "the helpers' formals, keyword arguments, internal defines, loop and match variables are bindings"
    (define bindings (helper-bindings))
    (for ([name (in-list '("crosstab" "labels" "width" "counts" "name" "column-values" "left"))])
      (check-not-false (member name bindings) name)))

  (test-case "a helper's binding that links on the helpers' page is a finding, elsewhere it is not"
    (define html "<a href=\"x.html\" class=\"RktValLink\" data-pltdoc=\"x\">type</a>")
    (check-equal? (html-findings (hash) '("type") '() helpers-page html)
                  '("examples.html: type, which a helper binds, links"))
    (check-equal? (html-findings (hash) '("type") '() "kmeans.html" html) '()))

  (test-case "a struct field the page declares is no name to link"
    (define field
      "<span class=\"RktSym\">scale</span><span class=\"hspace\">&nbsp;</span>:<span class=\"hspace\">&nbsp;</span>")
    (define prose "<span class=\"RktSym\">scale</span>")
    (define labelled (hash "scale" #t))
    (check-equal? (html-findings labelled '() '() "preprocessing.html" (string-append field prose))
                  '())
    (check-equal? (html-findings labelled '() '() "kmeans.html" prose)
                  '("kmeans.html: scale, which the examples can use, has no link")))

  (test-case "a module the examples require is no name to link, and links to no binding"
    (define modules (required-modules))
    (check-not-false (memq 'pict modules))
    (check-false (hash-ref (example-names modules) "pict" #f))
    (check-not-false (hash-ref (example-names modules) "pict-width" #f))
    (define html "<a href=\"x.html\" class=\"RktValLink\" data-pltdoc=\"x\">pict</a>")
    (check-equal?
     (html-findings (hash) '() '("pict") "getting-started.html" html)
     '("getting-started.html: pict, a module the examples require, links to a binding"))))
