#lang racket/base

(require (only-in racket/file find-files)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/path simple-form-path)
         (only-in racket/string string-join)
         (only-in "private/alpha.rkt" alpha-normal procedure-name)
         (only-in "private/imports.rkt" import-resolver)
         (only-in "private/modules.rkt" read-module rkt?))

(provide copied-definitions
         copied-definitions-in
         describe-copies)

(struct site (file line name) #:transparent)

(struct scope (language requires forms) #:transparent)

(define (struct-names name fields)
  (append (list name (symbol-append name '?) (symbol-append 'make- name))
          (for/list ([field (in-list fields)]
                     #:when (or (symbol? field) (and (pair? field) (symbol? (car field)))))
            (symbol-append name
                           '-
                           (if (pair? field)
                               (car field)
                               field)))))

(define (symbol-append . parts)
  (string->symbol (apply string-append (map symbol->string parts))))

(define (definer? head)
  (and (symbol? head) (regexp-match? #rx"^define" (symbol->string head))))

(define (defined-names datum)
  (match datum
    [(list* (or 'struct 'define-struct) (? symbol? name) (? symbol?) (? list? fields) _)
     (struct-names name fields)]
    [(list* (or 'struct 'define-struct) (? symbol? name) (? list? fields) _)
     (struct-names name fields)]
    [(list* 'define-values (? list? names) _) (filter symbol? names)]
    [(list* (? definer?) (? symbol? name) _) (list name)]
    [(list* (? definer?) (? pair? head) _) (list (head-name head))]
    [_ '()]))

(define (head-name head)
  (match head
    [(cons (? symbol? name) _) name]
    [(cons inner _) (head-name inner)]))

(define (requires-of forms)
  (filter (lambda (form)
            (match (syntax->datum form)
              [(cons 'require _) #t]
              [_ #f]))
          forms))

(define (module-scopes stx)
  (match (syntax-e stx)
    [(list _ _ language body)
     (define forms
       (match (syntax-e body)
         [(cons _ forms) forms]
         [_ '()]))
     (define top (scope (syntax->datum language) (requires-of forms) forms))
     (cons top (append-map (lambda (form) (submodule-scopes top form)) forms))]
    [_ '()]))

(define (submodule-scopes top form)
  (match (syntax-e form)
    [(list* (app syntax-e (or 'module+ 'module*)) _ (app syntax-e #f) forms)
     (list (scope (scope-language top) (append (scope-requires top) (requires-of forms)) forms))]
    [(list* (app syntax-e 'module*) _ language forms)
     (list (scope (syntax->datum language) (requires-of forms) forms))]
    [(list* (app syntax-e 'module+) _ forms)
     (list (scope (scope-language top) (append (scope-requires top) (requires-of forms)) forms))]
    [(list* (app syntax-e 'module) _ language forms)
     (list (scope (syntax->datum language) (requires-of forms) forms))]
    [_ '()]))

(define (definitions-in file)
  (define-values (directory _name _dir?) (split-path (simple-form-path file)))
  (define scopes (module-scopes (read-module file)))
  (define top-locals
    (match scopes
      [(cons top _) (append-map defined-names (map syntax->datum (scope-forms top)))]
      ['() '()]))
  (for*/list ([one (in-list scopes)]
              [locals (in-value (append top-locals
                                        (append-map defined-names
                                                    (map syntax->datum (scope-forms one)))))]
              [resolve (in-value (lazy-resolver directory one))]
              [form (in-list (scope-forms one))]
              [datum (in-value (syntax->datum form))]
              [name (in-value (procedure-name datum))]
              #:when name)
    (define (free symbol)
      (or (and (not (memq symbol locals)) (resolve symbol)) (format "~a@~a" symbol file)))
    (cons (alpha-normal datum name free) (site file (syntax-line form) name))))

(define (lazy-resolver directory one)
  (define resolve #f)
  (lambda (symbol)
    (unless resolve
      (set!
       resolve
       (import-resolver directory (scope-language one) (map syntax->datum (scope-requires one)))))
    (resolve symbol)))

(define (site<? a b)
  (define fa (path->string (site-file a)))
  (define fb (path->string (site-file b)))
  (or (string<? fa fb) (and (string=? fa fb) (< (site-line a) (site-line b)))))

(define (copied-definitions files)
  (define by-body (make-hash))
  (for* ([file (in-list files)]
         [entry (in-list (definitions-in file))])
    (match-define (cons body one) entry)
    (hash-update! by-body body (lambda (sites) (cons one sites)) '()))
  (sort (for*/list ([sites (in-hash-values by-body)]
                    [ordered (in-value (sort sites site<?))]
                    #:when (pair? (cdr (remove-duplicates (map site-file ordered)))))
          ordered)
        site<?
        #:key car))

(define (scanned? path)
  (and (rkt? path) (not (regexp-match? #rx"(^|/)info[.]rkt$" (path->string path)))))

(define (copied-definitions-in roots)
  (copied-definitions
   (sort (append-map (lambda (root) (find-files scanned? root)) roots) string<? #:key path->string)))

(define (describe-copies group)
  (string-join
   (for/list ([one (in-list group)])
     (format "~a:~a: ~a" (site-file one) (site-line one) (site-name one)))
   "\n  "
   #:before-first
   "copied definition: the same body after renaming, in more than one file; keep one, in the family's shared module (cuml-test's private/common-data.rkt when two families use it), and require it from there\n  "))

(module+ main
  (define groups
    (for*/list ([pool (in-vector (current-command-line-arguments))]
                [group (in-list (copied-definitions-in (regexp-split #rx":" pool)))])
      group))
  (for ([group (in-list groups)])
    (displayln (describe-copies group)))
  (unless (null? groups)
    (exit 1)))
