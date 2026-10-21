#lang racket/base

(require (only-in cuml/scribblings/guide/source page-helpers-source)
         (only-in racket/file file->string)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in racket/string string-replace))

(define raft-dtype-anchor #rx"lib\\._raft%2Farray\\.\\.rkt%29\\._dtype%29")

(define (exported-names module-path)
  (module-declared? module-path #t)
  (define-values (variables syntaxes) (module->exports module-path))
  (for*/list ([exports (in-list (list variables syntaxes))]
              [name+origins (in-list (match (assv 0 exports)
                                       [(cons _ names) names]
                                       [#f '()]))])
    (symbol->string (car name+origins))))

(define (example-names)
  (define-values (shown-require _helpers) (page-helpers-source))
  (match (syntax->datum shown-require)
    [(cons 'require module-paths)
     (for*/hash ([module-path (in-list (cons 'racket/base module-paths))]
                 [name (in-list (exported-names module-path))])
       (values name #t))]))

(define (unescaped html)
  (for/fold ([text html]) ([entity (in-list '(("&lt;" . "<") ("&gt;" . ">") ("&amp;" . "&")))])
    (string-replace text (car entity) (cdr entity))))

(define (unlinked-names html)
  (remove-duplicates
   (map unescaped
        (regexp-match* #rx"<span class=\"RktSym\">([^<]*)</span>" html #:match-select cadr))))

(define (page-findings labelled dir page)
  (define html (file->string (build-path dir page)))
  (define unlinked
    (for/list ([name (in-list (unlinked-names html))]
               #:when (hash-ref labelled name #f))
      (format "~a: ~a, which the examples can use, has no link" page name)))
  (cond
    [(and (equal? page "examples.html") (regexp-match? raft-dtype-anchor html))
     (cons (format "~a: a helper's local links to raft's dtype" page) unlinked)]
    [else unlinked]))

(module+ main
  (define dir
    (match (current-command-line-arguments)
      [(vector dir) dir]
      [_ (raise-user-error 'check-doc-links "usage: racket scripts/check-doc-links.rkt <html-dir>")]))
  (define labelled (example-names))
  (define findings
    (for*/list ([page (in-list (map path->string (directory-list dir)))]
                #:when (regexp-match? #rx"[.]html$" page)
                [finding (in-list (page-findings labelled dir page))])
      finding))
  (for-each displayln findings)
  (match findings
    ['() (printf "check-doc-links: every name the examples can use links in ~a\n" dir)]
    [_ (exit 1)]))
