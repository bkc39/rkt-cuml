#lang racket/base

(require (only-in racket/port port->string with-output-to-string)
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-false check-regexp-match check-true test-case)
         (only-in "../private/modules.rkt" with-temporary-directory))

(define-runtime-path gate "../../scripts/no-positional-access.sh")

(define-runtime-path comments-fixture "fixtures/positional-in-comments.rkt")

(define (gate-on forms)
  (gate-on-file (with-output-to-string (lambda ()
                                         (displayln "#lang racket/base")
                                         (for ([form (in-list forms)])
                                           (writeln form))))))

(define (gate-on-file text)
  (with-temporary-directory
   (root "positional-~a")
   (with-output-to-file (build-path root "table-test.rkt") (lambda () (display text)))
   (define ok? #f)
   (define output
     (with-output-to-string
      (lambda () (set! ok? (system* (find-executable-path "bash") gate (path->string root))))))
   (values ok? output)))

(test-case "a test table read with a positional accessor is refused, with what to use instead"
  (for ([accessor (in-list '(cadr caddr cadddr third fourth))])
    (define-values (ok? output) (gate-on `((define rows '((a 1 2 3))) (map ,accessor rows))))
    (check-false ok? (symbol->string accessor))
    (check-regexp-match
     (pregexp (format "table-test[.]rkt:3: ~a: read a test table's rows with match-define" accessor))
     output)))

(test-case "a table read with match-define passes, and so do names that only contain the accessor"
  (define-values (ok? output)
    (gate-on '((require (only-in racket/match match-define)) (define rows '((a 1 2)))
                                                             (for ([row (in-list rows)])
                                                               (match-define (list name x y) row)
                                                               (list name x y))
                                                             (define (caddr-free x)
                                                               x)
                                                             (define my-cadddr 4)
                                                             (define third-row 3))))
  (check-true ok?)
  (check-regexp-match #rx"no cadr, caddr, cadddr, third, fourth, or list-ref with a literal index"
                      output))

(test-case "list-ref with a literal index is refused, inside another access too"
  (define-values (ok? output) (gate-on '((define rows '((a 1 2 3))) (list-ref (list-ref rows 0) 2))))
  (check-false ok?)
  (check-regexp-match
   #rx"table-test[.]rkt:3: list-ref with a literal index: read a test table's rows with match-define"
   output)
  (check-equal? (length (regexp-match* #rx"list-ref with a literal index:" output)) 2))

(test-case "list-ref with a computed index passes"
  (define-values (ok? output)
    (gate-on '((define rows '((a 1 2 3))) (for/list ([i (in-range 3)])
                                            (list-ref rows (- 2 i))))))
  (check-true ok? output))

(test-case "an accessor named in a comment or a string is not a use"
  (define-values (ok? output) (gate-on-file (call-with-input-file comments-fixture port->string)))
  (check-true ok? output))
