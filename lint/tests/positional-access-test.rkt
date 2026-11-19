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
  (for ([accessor (in-list '(cadr caddr cadddr second third fourth fifth tenth))])
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
  (check-regexp-match
   #rx"no cadr, caddr, cadddr, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, or list-ref with a literal index"
   output))

(test-case "list-ref with a literal index is refused, inside another access too"
  (define-values (ok? output) (gate-on '((define rows '((a 1 2 3))) (list-ref (list-ref rows 0) 2))))
  (check-false ok?)
  (check-regexp-match
   #rx"table-test[.]rkt:3: list-ref with a literal index: read a test table's rows with match-define"
   output)
  (check-equal? (length (regexp-match* #rx"list-ref with a literal index:" output)) 2))

(test-case "list-ref with a quoted literal index is refused"
  (define-values (ok? output) (gate-on '((define rows '((a 1 2 3))) (list-ref rows '2))))
  (check-false ok?)
  (check-regexp-match
   #rx"table-test[.]rkt:3: list-ref with a literal index: read a test table's rows with match-define"
   output))

(test-case "list-ref with a computed index passes"
  (define-values (ok? output)
    (gate-on '((define rows '((a 1 2 3))) (for/list ([i (in-range 3)])
                                            (list-ref rows (- 2 i))))))
  (check-true ok? output))

(test-case "an accessor named in a comment or a string is not a use"
  (define-values (ok? output) (gate-on-file (call-with-input-file comments-fixture port->string)))
  (check-true ok? output))

(test-case "list-ref with a quasiquoted literal index is refused"
  (define-values (ok? output) (gate-on '((define rows '((a 1 2 3))) (list-ref rows `2))))
  (check-false ok?)
  (check-regexp-match
   #rx"table-test[.]rkt:3: list-ref with a literal index: read a test table's rows with match-define"
   output))

(for ([use (in-list '(((require racket/list) (define (scaled xs)
                                               (define tenth 0.1)
                                               (map (lambda (x) (* x tenth)) xs)))
                      ((define units '(second minute hour)))
                      ((define (timeout n)
                         (list n 'second)))
                      ((define header '(first second)))
                      ((define (pair-sum p)
                         (let-values ([(first second) (values (car p) (cdr p))])
                           (+ first second))))
                      ((define (weighted second xs)
                         (map (lambda (x) (* x second)) xs)))
                      ((define (row n)
                         `(second ,n)))
                      ((define (pairs xs)
                         (for*/list ([first (in-list xs)]
                                     [second (in-list xs)])
                           (cons first second))))
                      ((require racket/match) (define (gap row)
                                                (match-define (list* best second _) row)
                                                (- second best)))))])
  (test-case (format "a name the file binds, or quoted data, is not an access: ~s" use)
    (define-values (ok? output) (gate-on use))
    (check-true ok? output)))

(test-case "racket/list's accessor is refused beside quoted data, and inside an unquote"
  (for ([forms (in-list '(((require racket/list) (define units '(second minute))
                                                 (define (f xs)
                                                   (second xs)))
                          ((require racket/list) (define (f xs)
                                                   `(row ,(second xs))))))])
    (define-values (ok? output) (gate-on forms))
    (check-false ok? output)
    (check-regexp-match #rx"table-test[.]rkt:[0-9]+: second: read a test table's rows" output)))
