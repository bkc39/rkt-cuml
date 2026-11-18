#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/match match)
         (only-in racket/path find-relative-path)
         (only-in racket/string string-suffix?)
         (only-in syntax/modread with-module-reading-parameterization))

(provide describe-reference
         gpl-references)

(define (rkt? path)
  (define text (path->string path))
  (and (string-suffix? text ".rkt") (not (regexp-match? #rx"/compiled/" text))))

(define (collection-path root file)
  (regexp-replace #rx"[.]rkt$" (path->string (find-relative-path root file)) ""))

(define (read-module file)
  (with-input-from-file file
                        (lambda ()
                          (port-count-lines! (current-input-port))
                          (with-module-reading-parameterization read-syntax))))

(define (names-in stx)
  (match (syntax-e stx)
    [(? string? text) (list (cons (regexp-replace #rx"[.]rkt$" text "") stx))]
    [(? symbol? name) (list (cons (symbol->string name) stx))]
    [(? pair? items) (append-map names-in (syntax->list* items))]
    [_ '()]))

(define (syntax->list* items)
  (match items
    [(cons a d) (cons a (syntax->list* d))]
    ['() '()]
    [(? syntax? (app syntax-e (? pair? tail))) (syntax->list* tail)]
    [tail (list tail)]))

(define (gpl-references apache-root gpl-root gpl-packages)
  (define gpl
    (append gpl-packages
            (for/list ([file (in-list (find-files rkt? gpl-root))]
                       #:unless (regexp-match? #rx"(^|/)info[.]rkt$" (path->string file)))
              (collection-path gpl-root file))))
  (sort (remove-duplicates (for*/list ([file (in-list (find-files rkt? apache-root))]
                                       [named (in-list (names-in (read-module file)))]
                                       #:when (member (car named) gpl))
                             (list file (syntax-line (cdr named)) (car named))))
        string<?
        #:key (lambda (reference) (path->string (car reference)))))

(define (describe-reference reference)
  (match reference
    [(list file line name)
     (format
      "~a:~a: ~a is GPL; an Apache-2.0 package may not require it, so a helper both share lives in the Apache-2.0 package"
      file
      line
      name)]))

(module+ main
  (define arguments (vector->list (current-command-line-arguments)))
  (define references
    (match arguments
      [(list* apache gpl packages) (gpl-references apache gpl packages)]))
  (for-each (lambda (reference) (displayln (describe-reference reference))) references)
  (unless (null? references)
    (exit 1)))
