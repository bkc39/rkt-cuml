#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/port with-output-to-string)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-false check-regexp-match check-true test-case))

(define-runtime-path gate "../../scripts/no-positional-access.sh")

(define (gate-on forms)
  (define root (make-temporary-directory "positional-~a"))
  (dynamic-wind void
                (lambda ()
                  (with-output-to-file (build-path root "table-test.rkt")
                                       (lambda ()
                                         (displayln "#lang racket/base")
                                         (for-each writeln forms)))
                  (define ok? #f)
                  (define output
                    (with-output-to-string
                     (lambda ()
                       (set! ok? (system* (find-executable-path "bash") gate (path->string root))))))
                  (values ok? output))
                (lambda () (delete-directory/files root))))

(test-case "a test table read with caddr or cadddr is refused, with what to use instead"
  (for ([accessor (in-list '(caddr cadddr))])
    (define-values (ok? output)
      (gate-on `((define rows '((a 1 2 3)))
                 (map ,accessor rows))))
    (check-false ok?)
    (check-regexp-match #rx"table-test[.]rkt:3: read a test table's rows with match-define" output)))

(test-case "a table read with match-define passes, and so do names that only contain the accessor"
  (define-values (ok? output)
    (gate-on '((require (only-in racket/match match-define))
               (define rows '((a 1 2)))
               (for ([row (in-list rows)])
                 (match-define (list name x y) row)
                 (list name x y))
               (define (caddr-free x) x)
               (define my-cadddr 4))))
  (check-true ok?)
  (check-regexp-match #rx"no caddr or cadddr" output))
