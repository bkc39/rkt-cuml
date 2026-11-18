#lang racket/base

(require (only-in racket/match match))

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

;; A relative module path's index is shared between probes, and so is its
;; resolution: anchoring it keeps each file's "c.rkt" its own.
(define (anchored directory spec)
  (match spec
    [(? string? relative) `(file ,(absolute directory relative))]
    [(list 'file (? string? relative)) `(file ,(absolute directory relative))]
    [(list* (or 'lib 'planet 'quote) _) spec]
    [(list* 'submod (or "." "..") _) spec]
    [(? list?) (map (lambda (one) (anchored directory one)) spec)]
    [_ spec]))

(define (expand-probe directory language requires)
  (call-in directory
           (lambda ()
             (find-marker (expand (datum->syntax #f
                                                 `(module probe ,(anchored directory language)
                                                    ,@(anchored directory requires)
                                                    (quote-syntax ,marker))))))))

(define (expands? directory language requires)
  (with-handlers ([exn:fail? (lambda (_failure) #f)])
    (expand-probe directory language requires)))

(define (refuse file directory language requires failure)
  (define culprit
    (for*/first ([form (in-list requires)]
                 [spec (in-list (cdr form))]
                 #:unless (expands? directory language (list `(require ,spec))))
      spec))
  (error
   'copied-definitions
   "~a: cannot resolve the names it imports, so its definitions cannot be compared; fix ~s or teach lint/private/imports.rkt to read it\n  ~a"
   file
   (or culprit
       `(module _ ,language
          ,@requires))
   (exn-message failure)))

(define (probe-identifier file directory language requires)
  (with-handlers ([exn:fail? (lambda (failure) (refuse file directory language requires failure))])
    (expand-probe directory language requires)))

(define (binding-key binding)
  (match binding
    [(list* source name _)
     (format "~s:~a" (resolved-module-path-name (module-path-index-resolve source)) name)]
    [_ #f]))

(define (import-resolver file language requires)
  (define-values (directory _name _dir?) (split-path (path->complete-path file)))
  (define here (probe-identifier file directory language requires))
  (define keys (make-hasheq))
  (lambda (name)
    (hash-ref! keys
               name
               (lambda ()
                 (call-in directory
                          (lambda ()
                            (binding-key (identifier-binding (datum->syntax here name) 0))))))))
