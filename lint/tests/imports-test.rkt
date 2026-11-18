#lang racket/base

(require (only-in racket/port with-output-to-string)
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
                                   '((require (only-in "c.rkt" first)))))
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
