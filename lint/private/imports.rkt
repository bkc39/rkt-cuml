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

(define (probe-identifier directory language requires)
  (with-handlers ([exn:fail? (lambda (_failure) #f)])
    (call-in directory
             (lambda ()
               (find-marker (expand (datum->syntax #f
                                                   `(module probe ,language
                                                      ,@requires
                                                      (quote-syntax ,marker)))))))))

(define (binding-key binding)
  (match binding
    [(list* source name _)
     (format "~s:~a" (resolved-module-path-name (module-path-index-resolve source)) name)]
    [_ #f]))

(define (import-resolver directory language requires)
  (define here (probe-identifier directory language requires))
  (define keys (make-hasheq))
  (lambda (name)
    (and here
         (hash-ref! keys
                    name
                    (lambda ()
                      (call-in directory
                               (lambda ()
                                 (binding-key (identifier-binding (datum->syntax here name) 0)))))))))
