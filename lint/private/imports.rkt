#lang racket/base

(require (only-in racket/match match match* match-define match-lambda))

(provide import-resolver)

(define marker 'cuml-lint-imports-here)

(define namespace (make-base-namespace))

(define (find-marker one)
  (cond
    [(syntax? one)
     (if (eq? (syntax-e one) marker)
         one
         (find-marker (syntax-e one)))]
    [(pair? one) (or (find-marker (car one)) (find-marker (cdr one)))]
    [else #f]))

(define (call-in directory proc)
  (parameterize ([current-namespace namespace]
                 [current-load-relative-directory directory]
                 [current-directory directory])
    (proc)))

(define (absolute directory relative)
  (path->string (simplify-path (path->complete-path relative directory))))

(define (anchored file here spec)
  (define-values (directory _name _dir?) (split-path file))
  (define (module-path one)
    (anchored file here one))
  (match spec
    [(? string? relative) `(file ,(absolute directory relative))]
    [(list 'file (? string? relative)) `(file ,(absolute directory relative))]
    [(list 'quote (? symbol? name))
     #:when (not (regexp-match? #rx"^#%" (symbol->string name)))
     (own-submodule file here (list "." name) spec)]
    [(list* 'submod (and root (or "." "..")) path) (own-submodule file here (cons root path) spec)]
    [(list* 'submod root path) (list* 'submod (module-path root) path)]
    [(list* (and head (or 'only-in 'except-in 'rename-in)) inner names)
     (list* head (module-path inner) names)]
    [(list (and head (or 'prefix-in 'filtered-in 'matching-identifiers-in)) argument inner)
     (list head argument (module-path inner))]
    [(list* (and head (or 'combine-in 'for-syntax 'for-template 'for-label 'subtract-in)) specs)
     (cons head (map module-path specs))]
    [(list* (and head (or 'for-meta 'only-meta-in 'for-space 'only-space-in)) argument specs)
     (list* head argument (map module-path specs))]
    [(list* 'multi-in subs) (multi-in file here subs spec)]
    [(cons 'relative-in _) (unanchored spec)]
    [(cons (or 'lib 'planet) _) spec]
    [(? holds-string?) (unanchored spec)]
    [_ spec]))

(define (holds-string? datum)
  (match datum
    [(? string?) #t]
    [(cons a d) (or (holds-string? a) (holds-string? d))]
    [_ #f]))

(define (unanchored spec)
  (error 'copied-definitions
         "~s names a module by a relative path inside a form the lint cannot anchor to its file"
         spec))

(define (own-submodule file here path spec)
  (define names
    (for/fold ([names (reverse here)]) ([step (in-list path)])
      (match* (step names)
        [(_ #f) #f]
        [("." _) names]
        [(".." (cons _ outer)) outer]
        [((? symbol? name) _) (cons name names)]
        [(_ _) #f])))
  (match names
    [#f spec]
    ['() `(file ,(path->string file))]
    [_ `(submod (file ,(path->string file)) ,@(reverse names))]))

(define (multi-in file here subs spec)
  (define paths
    (for/fold ([paths '("")]) ([sub (in-list subs)])
      (for*/list ([prefix (in-list paths)]
                  [part (in-list (if (list? sub)
                                     sub
                                     (list sub)))])
        (and prefix (string? part) (string-append prefix (if (equal? prefix "") "" "/") part)))))
  (cond
    [(andmap string? paths) (cons 'combine-in (map (lambda (one) (anchored file here one)) paths))]
    [(holds-string? spec) (unanchored spec)]
    [else spec]))

(define (expand-probe given language requires)
  (define file (path->complete-path given))
  (define-values (directory _name _dir?) (split-path file))
  (call-in directory
           (lambda ()
             (find-marker
              (expand (datum->syntax
                       #f
                       `(module probe ,(anchored file '() language)
                          ,@(for/list ([one (in-list requires)])
                              (match-define (list here (cons head specs)) one)
                              (cons head (map (lambda (spec) (anchored file here spec)) specs)))
                          (quote-syntax ,marker))))))))

(define (expands? file language requires)
  (with-handlers ([exn:fail? (lambda (_failure) #f)])
    (expand-probe file language requires)))

(define (refuse file language requires failure)
  (define culprit
    (for*/first ([one (in-list requires)]
                 [spec (in-list (match one
                                  [(list _ (cons _ specs)) specs]))]
                 #:unless (expands? file
                                    language
                                    (match one
                                      [(list here _) (list (list here `(require ,spec)))])))
      spec))
  (error
   'copied-definitions
   "~a: cannot resolve the names it imports, so its definitions cannot be compared; fix ~s or teach lint/private/imports.rkt to read it\n  ~a"
   file
   (or culprit
       `(module _ ,language
          ,@(map (match-lambda
                   [(list _ form) form])
                 requires)))
   (exn-message failure)))

(define (probe-identifier file language requires)
  (with-handlers ([exn:fail? (lambda (failure) (refuse file language requires failure))])
    (expand-probe file language requires)))

(define (binding-key binding)
  (match binding
    [(list* source name _)
     (format "~s:~a" (resolved-module-path-name (module-path-index-resolve source)) name)]
    [_ #f]))

(define (import-resolver file language requires)
  (define-values (directory _name _dir?) (split-path (path->complete-path file)))
  (define here (probe-identifier file language requires))
  (define keys (make-hasheq))
  (lambda (name)
    (hash-ref! keys
               name
               (lambda ()
                 (call-in directory
                          (lambda ()
                            (binding-key (identifier-binding (datum->syntax here name) 0))))))))
