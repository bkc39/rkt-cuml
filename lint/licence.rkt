#lang racket/base

(require (only-in pkg/name package-source->name)
         (only-in racket/file find-files)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/path find-relative-path simple-form-path)
         (only-in racket/string string-join string-prefix?)
         (only-in "private/modules.rkt" read-module rkt?))

(provide describe-reference
         gpl-references)

(struct reference (stx name path) #:transparent)

(define (collection-path root file)
  (regexp-replace #rx"[.]rkt$" (path->string (find-relative-path root file)) ""))

(define (unsuffixed text)
  (regexp-replace #rx"[.](rkt|ss|scrbl)$" text ""))

(define (named stx text)
  (list (reference stx (unsuffixed text) #f)))

(define (pathed stx text)
  (list (reference stx text text)))

(define (items stx)
  (match (syntax->list stx)
    [(? list? forms) forms]
    [_ '()]))

(define (head-of stx)
  (match (items stx)
    [(cons (app syntax-e (? symbol? head)) _) head]
    [_ #f]))

(define (arguments-of stx)
  (match (items stx)
    [(cons _ arguments) arguments]
    ['() '()]))

(define (spec-references stx)
  (define arguments (arguments-of stx))
  (match (syntax-e stx)
    [(? symbol? name) (named stx (symbol->string name))]
    [(? string? text) (pathed stx text)]
    [_
     (match (head-of stx)
       ['lib
        (match arguments
          [(list (app syntax-e (? string? text))) (named stx (lib-name text))]
          [(list* (app syntax-e (? string? file)) collections)
           (named stx (string-join (append (map syntax-e collections) (list file)) "/"))]
          [_ '()])]
       ['file
        (match arguments
          [(list (app syntax-e (? string? text))) (pathed stx text)]
          [_ '()])]
       ['quote (append-map spec-references arguments)]
       [(or 'only-in 'except-in 'rename-in 'submod)
        (match arguments
          [(cons spec _) (spec-references spec)]
          [_ '()])]
       [(or 'only-space-in 'only-meta-in)
        (match arguments
          [(list* _ specs) (append-map spec-references specs)]
          [_ '()])]
       ['prefix-in
        (match arguments
          [(list _ spec) (spec-references spec)]
          [_ '()])]
       [(or 'for-syntax 'for-template 'for-label 'combine-in 'relative-in)
        (append-map spec-references arguments)]
       [(or 'for-meta 'for-space)
        (match arguments
          [(cons _ specs) (append-map spec-references specs)]
          [_ '()])]
       [_ '()])]))

(define (lib-name text)
  (define name (unsuffixed text))
  (if (regexp-match? #rx"/" name)
      name
      (string-append name "/main")))

(define (collection-file-references stx arguments)
  (define strings
    (for/list ([argument (in-list arguments)]
               #:break (keyword? (syntax-e argument)))
      (syntax-e argument)))
  (match strings
    [(cons (? string? file) collections)
     #:when (andmap string? collections)
     (named stx (string-join (append collections (list file)) "/"))]
    [_ '()]))

(define (module-references stx)
  (define arguments (arguments-of stx))
  (match (head-of stx)
    [(or 'quote 'quasiquote 'quote-syntax 'syntax 'quasisyntax) '()]
    ['require (append-map spec-references arguments)]
    ['dynamic-require
     (match arguments
       [(cons spec _) (append (spec-references spec) (module-references spec))]
       [_ '()])]
    ['collection-file-path (collection-file-references stx arguments)]
    [(or 'module 'module*)
     (match arguments
       [(list* _ language body)
        (append (spec-references language) (append-map module-references body))]
       [_ '()])]
    [_ (append-map module-references (items stx))]))

(define (dependency-references stx)
  (match (syntax->datum stx)
    [(list 'define (or 'deps 'build-deps 'implies) (list 'quote (? list? entries)))
     (for/list ([entry (in-list entries)]
                #:when (or (string? entry) (and (pair? entry) (string? (car entry)))))
       (define source
         (if (pair? entry)
             (car entry)
             entry))
       (reference stx (or (package-source->name source) source) #f))]
    [_ '()]))

(define (info? file)
  (regexp-match? #rx"(^|/)info[.]rkt$" (path->string file)))

(define (references-in file)
  (define stx (read-module file))
  (if (info? file)
      (append-map dependency-references (module-body stx))
      (module-references stx)))

(define (module-body stx)
  (match (items stx)
    [(list _ _ _ (app items (cons _ forms))) forms]
    [_ '()]))

(define (under? path root)
  (string-prefix? (path->string path) (path->string (path->directory-path root))))

(define (gpl-reference? found file gpl-root gpl-names gpl-packages)
  (match found
    [(reference _ _ (? string? text))
     (define-values (directory _name _dir?) (split-path file))
     (define target (simple-form-path (path->complete-path text (path->complete-path directory))))
     (or (under? target gpl-root)
         (for/or ([element (in-list (explode-path target))])
           (member (path->string element) gpl-packages)))]
    [(reference _ name #f)
     (or (member name gpl-names)
         (member (string-append name "/main") gpl-names)
         (for/or ([package (in-list gpl-packages)])
           (or (string=? name package) (string-prefix? name (string-append package "/")))))]))

(define (gpl-references apache-root gpl-root gpl-packages)
  (define gpl-directory (simple-form-path gpl-root))
  (define gpl-names
    (for/list ([file (in-list (find-files rkt? gpl-root))]
               #:unless (info? file))
      (collection-path gpl-root file)))
  (sort (remove-duplicates
         (for*/list ([file (in-list (find-files rkt? apache-root))]
                     [found (in-list (references-in file))]
                     #:when (gpl-reference? found file gpl-directory gpl-names gpl-packages))
           (list file (syntax-line (reference-stx found)) (reference-name found))))
        string<?
        #:key (lambda (one) (path->string (car one)))))

(define (describe-reference one)
  (match-define (list file line name) one)
  (format
   "~a:~a: ~a is GPL; an Apache-2.0 package may not require it, so a helper both share lives in the Apache-2.0 package"
   file
   line
   name))

(module+ main
  (match-define (list* apache gpl packages) (vector->list (current-command-line-arguments)))
  (define references (gpl-references apache gpl packages))
  (for ([one (in-list references)])
    (displayln (describe-reference one)))
  (unless (null? references)
    (exit 1)))
