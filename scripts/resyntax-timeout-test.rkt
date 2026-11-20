#lang racket/base

(require (only-in racket/port with-output-to-string)
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-false check-regexp-match check-true test-case)
         (only-in "../lint/private/modules.rkt" with-temporary-directory write-module))

(define-runtime-path resyntax-script "resyntax.sh")

(define-runtime-path fake-resyntax "fixtures/fake-resyntax")

(define fixture
  '((define (double x)
      (* 2 x))
    (double 21)))

(define (call-with-fixture proc)
  (with-temporary-directory (root "resyntax-timeout-~a")
                            (define file (build-path root "module.rkt"))
                            (write-module file "#lang racket/base" fixture)
                            (proc file)))

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
     (check-regexp-match
      #rx"\n  [^\n]*resyntax-timeout-[0-9]+/module[.]rkt: [^\n]*identifier-usage-analyzer"
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

(define (run-faked file rc #:pad-lines [pad-lines 0])
  (parameterize ([current-environment-variables
                  (environment-variables-copy (current-environment-variables))])
    (putenv "PATH" (format "~a:~a" (path->string fake-resyntax) (getenv "PATH")))
    (putenv "FAKE_RESYNTAX_FILE" (path->string file))
    (putenv "FAKE_RESYNTAX_RC" (number->string rc))
    (putenv "FAKE_RESYNTAX_PAD_LINES" (number->string pad-lines))
    (run-on file)))

(test-case "a file whose path has a space is listed whole, with every analyzer it missed"
  (with-temporary-directory
   (root "resyntax-space-~a")
   (define file (build-path root "a module.rkt"))
   (write-module file "#lang racket/base" fixture)
   (define-values (ok? output) (run-faked file 0))
   (check-true ok?)
   (check-regexp-match
    #rx"\n  [^\n]*/a module[.]rkt: identifier-usage-analyzer, variable-mutability-analyzer\n"
    output)))

(test-case "the files an analyzer missed are listed when Resyntax itself fails"
  (call-with-fixture (lambda (file)
                       (define-values (ok? output) (run-faked file 3))
                       (check-false ok?)
                       (check-regexp-match #rx"files not fully analysed: 1;" output)
                       (check-regexp-match #rx"resyntax exited 3" output))))

(test-case "analyzer output over the 128 KiB argument limit is listed, and the lane still passes"
  (call-with-fixture
   (lambda (file)
     (define-values (ok? output) (run-faked file 0 #:pad-lines 4000))
     (check-true (> (string-length output) 300000))
     (check-true ok?)
     (check-false (regexp-match? #rx"Argument list too long" output))
     (check-regexp-match #rx"files not fully analysed: 1;" output)
     (check-regexp-match
      #rx"\n  [^\n]*/module[.]rkt: identifier-usage-analyzer, variable-mutability-analyzer\n"
      output)
     (check-regexp-match #rx"in the analysed files; files not fully analysed: 1," output))))
