#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/port with-output-to-string)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-false check-regexp-match check-true test-case))

(define-runtime-path resyntax-script "resyntax.sh")

(define fixture '((define (double x) (* 2 x)) (double 21)))

(define (call-with-fixture proc)
  (define root (make-temporary-directory "resyntax-timeout-~a"))
  (define file (build-path root "module.rkt"))
  (dynamic-wind void
                (lambda ()
                  (with-output-to-file file
                                       (lambda ()
                                         (displayln "#lang racket/base")
                                         (for-each writeln fixture)))
                  (proc file))
                (lambda () (delete-directory/files root))))

(define (run-on file . options)
  (define ok? #f)
  (define output
    (with-output-to-string (lambda ()
                             (parameterize ([current-error-port (current-output-port)])
                               (set! ok?
                                     (apply system*
                                            (find-executable-path "bash")
                                            resyntax-script
                                            (append options (list "--file" (path->string file)))))))))
  (values ok? output))

(test-case "an analyzer timeout is listed by file as a warning, and the lane still passes"
  (call-with-fixture
   (lambda (file)
     (define-values (ok? output) (run-on file "--analyzer-timeout" "1"))
     (check-true ok?)
     (check-regexp-match #rx"warning: files not fully analysed: 1;" output)
     (check-regexp-match #rx"\n  [^\n]*resyntax-timeout-[0-9]+/module[.]rkt: [^\n]*identifier-usage-analyzer"
                         output)
     (check-regexp-match #rx"in the analysed files; files not fully analysed: 1," output)
     (check-false (regexp-match? #rx"every file fully analysed" output)))))

(test-case "RESYNTAX_ANALYZER_TIMEOUT sets the analyzer timeout"
  (call-with-fixture
   (lambda (file)
     (define-values (ok? output)
       (parameterize ([current-environment-variables
                       (environment-variables-copy (current-environment-variables))])
         (putenv "RESYNTAX_ANALYZER_TIMEOUT" "1")
         (run-on file)))
     (check-true ok?)
     (check-regexp-match #rx"analyzer timeout 1 ms" output)
     (check-regexp-match #rx"warning: files not fully analysed: 1;" output))))

(test-case "a run that analyses every file says so"
  (call-with-fixture (lambda (file)
                       (define-values (ok? output) (run-on file))
                       (check-true ok?)
                       (check-regexp-match #rx"every file fully analysed" output)
                       (check-false (regexp-match? #rx"not fully analysed" output)))))

(test-case "an analyzer timeout that is not a number of milliseconds is refused"
  (call-with-fixture (lambda (file)
                       (define-values (ok? output) (run-on file "--analyzer-timeout" "30s"))
                       (check-false ok?)
                       (check-regexp-match #rx"whole number of milliseconds, not '30s'" output))))
