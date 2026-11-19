#lang racket/base

(require (only-in racket/match match-define)
         (only-in racket/port with-output-to-string)
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-exn check-regexp-match check-true test-case)
         (only-in "../duplicates.rkt" copied-definitions-in)
         (only-in "../private/modules.rkt" with-temporary-directory write-module))

(define-runtime-path imports "../private/imports.rkt")
(define-runtime-path modules "../private/modules.rkt")

(define (probe-program imports-path modules-path)
  `((require (only-in (file ,imports-path) import-resolver)
             (only-in (file ,modules-path) with-temporary-directory write-module))
    (define (stale-count)
      (for/sum ([_ (in-range 12)])
               (with-temporary-directory
                (root "imports-~a")
                (write-module (build-path root "c.rkt")
                              "#lang racket/base"
                              '((provide first) (define (first xs)
                                                  xs)))
                (define resolve
                  (import-resolver (build-path root "a.rkt")
                                   'racket/base
                                   '((() (require (only-in "c.rkt" first))))))
                (if (equal? (resolve 'first) (format "~s:first" (build-path root "c.rkt"))) 0 1))))
    (provide stale-count)
    (module+ test
      (require (only-in rackunit check-equal?))
      (check-equal? (stale-count) 0))))

(test-case "a relative require resolves against its own file, however many directories came before"
  (parameterize ([current-namespace (make-base-namespace)])
    (eval `(module probe racket/base
             ,@(probe-program (path->string imports) (path->string modules))))
    (namespace-require ''probe)
    (check-equal? (eval '(stale-count)) 0)))

(test-case "the same holds under raco test --process"
  (with-temporary-directory
   (root "imports-process-~a")
   (define file (build-path root "probe-test.rkt"))
   (write-module file
                 "#lang racket/base"
                 (probe-program (path->string imports) (path->string modules)))
   (define ok? #f)
   (define output
     (with-output-to-string (lambda ()
                              (parameterize ([current-error-port (current-output-port)])
                                (set! ok?
                                      (system* (find-executable-path (find-system-path 'exec-file))
                                               "-l-"
                                               "raco"
                                               "test"
                                               "--process"
                                               (path->string file)))))))
   (check-true ok? output)
   (check-regexp-match #rx"1 test passed" output)))

(test-case "a require that does not resolve fails the lint, naming the file and the require"
  (with-temporary-directory
   (root "imports-broken-~a")
   (write-module (build-path root "a.rkt")
                 "#lang racket/base"
                 '((require racket/list
                            (only-in "missing.rkt" first))
                   (define (head xs)
                     (first xs))))
   (check-exn (lambda (failure)
                (and (exn:fail? failure)
                     (regexp-match? #rx"a[.]rkt: cannot resolve the names it imports"
                                    (exn-message failure))
                     (regexp-match? (regexp-quote "fix (only-in \"missing.rkt\" first)")
                                    (exn-message failure))))
              (lambda () (copied-definitions-in (list root))))))

(define (copies-in files)
  (with-temporary-directory (root "imports-cases-~a")
                            (for ([module (in-list files)])
                              (match-define (list name forms) module)
                              (write-module (build-path root name) "#lang racket/base" forms))
                            (for/list ([group (in-list (copied-definitions-in (list root)))])
                              (length group))))

(define helpers
  '(module helpers racket/base
     (provide h)
     (define (h)
       1)))

(for ([way (in-list '("'m" "(submod \".\" m)"
                           "(submod \"..\" m) in module+ main"
                           "(submod \".\" m) in module+ test"))]
      [forms (in-list `((,helpers (require 'helpers)
                                  (define (f)
                                    (h)))
                        (,helpers (require (submod "." helpers))
                                  (define (f)
                                    (h)))
                        (,helpers (module+ main
                                    (require (submod ".." helpers))
                                    (define (f)
                                      (h))))
                        ((module+ test
                           ,helpers
                           (require (only-in (submod "." helpers) h))
                           (define (f)
                             (h))))))])
  (test-case (format "a file that requires its own submodule by ~a is read through the file" way)
    (check-equal? (copies-in `(("a.rkt" ,forms))) '())))

(test-case "an own submodule's names resolve to that submodule"
  (check-equal? (copies-in `(("a.rkt" (,helpers (require 'helpers)
                                                (define (f)
                                                  (h))))
                             ("b.rkt" ((require (submod "a.rkt" helpers)) (define (g)
                                                                            (h))))))
                '(2)))

(test-case "only module paths are anchored, not the strings a filter reads"
  (check-equal? (copies-in '(("c.rkt" ((provide first) (define (first xs)
                                                         xs)))
                             ("a.rkt" ((require racket/require
                                                (for-syntax racket/base
                                                            (only-in racket/string string-prefix?))
                                                (filtered-in (lambda (name)
                                                               (and (string-prefix? name "fir") name))
                                                             "c.rkt"))
                                       (define (f xs)
                                         (first xs))))
                             ("b.rkt" ((require (only-in "c.rkt" first)) (define (g xs)
                                                                           (first xs))))))
                '(2)))

(test-case "the strings of a multi-in join to module paths before they are anchored"
  (check-equal? (copies-in '(("c.rkt" ((provide first) (define (first xs)
                                                         xs)))
                             ("a.rkt" ((require racket/require
                                                (multi-in "." ("c.rkt")))
                                       (define (f xs)
                                         (first xs))))
                             ("b.rkt" ((require (only-in "c.rkt" first)) (define (g xs)
                                                                           (first xs))))))
                '(2)))

(test-case "a module+ that imports a name its file already imports fails the lint loudly"
  (check-exn #rx"a[.]rkt: cannot resolve the names it imports"
             (lambda ()
               (copies-in '(("c.rkt" ((provide first) (define (first xs)
                                                        xs)))
                            ("a.rkt" ((require racket/list) (module+ test
                                                              (require (only-in "c.rkt" first))
                                                              (define (f xs)
                                                                (first xs))))))))))
