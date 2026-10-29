#lang racket/base

(require (only-in cuml/scribblings/guide/source page-helpers-path page-helpers-source)
         (only-in racket/dict in-dict)
         (only-in racket/list append* append-map remove-duplicates)
         (only-in racket/match == match)
         (only-in racket/string string-replace)
         (only-in syntax/parse/pre ~or* attribute id kernel-literals syntax-parse)
         (only-in threading ~>))

(define helpers-page "examples.html")

(define linked-name #rx"<a [^>]*class=\"Rkt[A-Za-z]*Link\"[^>]*>([^<]*)</a>")

(define unlinked-name #rx"<span class=\"RktSym\">([^<]*)</span>")

(define field-declaration
  #rx"<span class=\"RktSym\">([^<]*)</span><span class=\"hspace\">&nbsp;</span>:")

(define definition-box #rx"<table [^>]*class=\"boxed RBoxed\"")

(define table-tag #rx"<(/)?table[ >]")

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

(define (table-end html start)
  (let loop ([position start]
             [depth 0])
    (match (regexp-match-positions table-tag html position)
      [(list (cons _ tag-end) #f) (loop tag-end (add1 depth))]
      [(list (cons _ tag-end) _)
       (match depth
         [1 tag-end]
         [_ (loop tag-end (sub1 depth))])]
      [#f (string-length html)])))

(define (field-scopes html)
  (let loop ([position 0])
    (match (regexp-match-positions definition-box html position)
      [(list (cons start _))
       (define end (table-end html start))
       (define box (substring html start end))
       (list* (cons (substring html position start) '())
              (cons box (rendered-names field-declaration box))
              (loop end))]
      [#f (list (cons (substring html position) '()))])))

(define (unlinked-names html)
  (remove-duplicates (for*/list ([(text fields) (in-dict (field-scopes html))]
                                 [name (in-list (rendered-names unlinked-name text))]
                                 #:unless (member name fields))
                       name)))

(define (html-findings labelled bindings modules page html)
  (define shown-bindings
    (match page
      [(== helpers-page) bindings]
      [_ '()]))
  (append (for/list ([name (in-list (rendered-names linked-name html))]
                     #:when (member name shown-bindings))
            (format "~a: ~a, which a helper binds, links" page name))
          (for/list ([name (in-list (rendered-names linked-name html))]
                     #:when (member name modules))
            (format "~a: ~a, a module the examples require, links to a binding" page name))
          (for/list ([name (in-list (unlinked-names html))]
                     #:when (hash-ref labelled name #f))
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
    (for ([name (in-list
                 '("matched-labels" "labels" "width" "overlap" "name" "column-values" "left"))])
      (check-not-false (member name bindings) name)))

  (test-case "a helper's binding that links on the helpers' page is a finding, elsewhere it is not"
    (define html "<a href=\"x.html\" class=\"RktValLink\" data-pltdoc=\"x\">type</a>")
    (check-equal? (html-findings (hash) '("type") '() helpers-page html)
                  '("examples.html: type, which a helper binds, links"))
    (check-equal? (html-findings (hash) '("type") '() "kmeans.html" html) '()))

  (define (symbol-cell name)
    (format "<span class=\"RktSym\">~a</span>" name))
  (define (declaration name)
    (string-append (symbol-cell name)
                   "<span class=\"hspace\">&nbsp;</span>:<span class=\"hspace\">&nbsp;</span>"))
  (define (rows cell names)
    (apply string-append
           (for/list ([name (in-list names)])
             (format "<tr><td>~a</td></tr>" (cell name)))))
  (define (rendered-box header . fields)
    (format
     (string-append "<table cellspacing=\"0\" cellpadding=\"0\" class=\"boxed RBoxed\">"
                    "<tr><td><blockquote class=\"SubFlow\">~a</blockquote></td></tr>~a</table>")
     header
     (rows declaration fields)))
  (define (struct-header first-field . fields)
    (format (string-append "<table cellspacing=\"0\" cellpadding=\"0\" class=\"RForeground\">"
                           "<tr><td><span class=\"nobreak\">~a</span></td></tr>~a</table>")
            (symbol-cell first-field)
            (rows symbol-cell fields)))
  (define labelled (hash "scale" #t))
  (define (scale-findings page html)
    (html-findings labelled '() '() page html))

  (test-case "a struct header's first and last fields are no names to link"
    (check-equal?
     (scale-findings "preprocessing.html"
                     (string-append (rendered-box (struct-header "scale" "mean") "scale" "mean")
                                    (rendered-box (struct-header "mean" "scale") "mean" "scale")))
     '()))

  (test-case "a one-line struct header's fields are no names to link"
    (define header
      (string-append "<p class=\"RForeground\"><span class=\"RktPn\">(</span>"
                     (symbol-cell "scale")
                     "<span class=\"RktPn\">)</span></p>"))
    (check-equal? (scale-findings "preprocessing.html" (rendered-box header "scale")) '()))

  (test-case "a bare identifier on a code-block line is a name to link"
    (define block
      (format
       "<table cellspacing=\"0\" cellpadding=\"0\" class=\"RktBlk\"><tr><td>~a</td></tr></table>"
       (symbol-cell "scale")))
    (check-equal? (scale-findings "kmeans.html" block)
                  '("kmeans.html: scale, which the examples can use, has no link")))

  (test-case "a field's name is skipped in its own definition box only"
    (check-equal? (scale-findings "preprocessing.html"
                                  (string-append (rendered-box (struct-header "scale") "scale")
                                                 (symbol-cell "scale")))
                  '("preprocessing.html: scale, which the examples can use, has no link"))
    (check-equal? (scale-findings "preprocessing.html" (rendered-box (symbol-cell "scale") "mean"))
                  '("preprocessing.html: scale, which the examples can use, has no link"))
    (check-equal? (scale-findings "kmeans.html" (declaration "scale"))
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
