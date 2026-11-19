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

(for ([use (in-list '(((require racket/list) (define (keep rows)
                                               (for/list ([row (in-list rows)]
                                                          #:when (second row))
                                                 row)))
                      ((define (keep rows)
                         (for/list ([row (in-list rows)]
                                    #:unless (cadr row))
                           row)))
                      ((require racket/list) (define (f rows)
                                               (for/fold ([acc '()]
                                                          #:result (second acc))
                                                         ([r rows])
                                                 (cons r acc))))
                      ((require racket/list racket/match) (define (f rows)
                                                            (match-define (list (app second x)) rows)
                                                            x))
                      ((require racket/list racket/match) (define (f row)
                                                            (match-define `(x second ,y) row)
                                                            (second y)))))])
  (test-case (format "a guard, a result, or a pattern's operator or literal is an access: ~s" use)
    (define-values (ok? output) (gate-on use))
    (check-false ok? output)
    (check-regexp-match #rx"table-test[.]rkt:[0-9]+: (second|cadr): read a test table's rows" output)))

(for ([use (in-list '(((require racket/match) (define (gap row)
                                                (match row
                                                  [(list best second) (- second best)])))
                      ((require racket/match) (define (gap row other)
                                                (match* (row other)
                                                  [((list best second) _) (- second best)])))
                      ((require racket/match) (define gap
                                                (match-lambda
                                                  [(list best second) (- second best)])))
                      ((require racket/match) (define (gap row)
                                                (match-let ([(list best second) row])
                                                  (- second best))))
                      ((require racket/match) (define (gap row)
                                                (match-let* ([(list best second) row])
                                                  (- second best))))
                      ((struct span (first second)) (define (len s)
                                                      (- (span-second s) (span-first s))))
                      ((define (f p)
                         (define-values (first second) (values (car p) (cdr p)))
                         (+ first second)))
                      ((define (f p)
                         (let* ([first (car p)]
                                [second (cdr p)])
                           (+ first second))))))])
  (test-case (format "a match pattern or a struct field that binds the name is not an access: ~s" use)
    (define-values (ok? output) (gate-on use))
    (check-true ok? output)))

(test-case "a guard's accessor does not hide the same accessor elsewhere in the file"
  (define-values (ok? output)
    (gate-on '((require racket/list) (define (keep rows)
                                       (for/list ([row (in-list rows)]
                                                  #:when (second row))
                                         row))
                                     (define (col rows)
                                       (map second rows)))))
  (check-false ok? output)
  (check-regexp-match #rx"table-test[.]rkt:3: second: read a test table's rows" output)
  (check-regexp-match #rx"table-test[.]rkt:4: second: read a test table's rows" output))

(test-case "racket/list's accessor is refused beside a match that binds another name"
  (define-values (ok? output)
    (gate-on '((require racket/list racket/match) (define (gap row)
                                                    (match row
                                                      [(list best worst) (- worst best)]))
                                                  (define (g xs)
                                                    (second xs)))))
  (check-false ok? output)
  (check-regexp-match #rx"table-test[.]rkt:4: second: read a test table's rows" output))

(test-case "an unquote inside a nested quasiquote is data"
  (define-values (ok? output)
    (gate-on '((require racket/list) (define (tmpl xs)
                                       `(a `(b ,(second xs)))))))
  (check-true ok? output))

(for ([use (in-list '(((require racket/list) (define (tmpl xs)
                                               `(a `(b ,,(second xs)))))
                      ((require racket/list) (define (tmpl xs)
                                               `#&,(second xs)))
                      ((require racket/list) (define (tmpl xs)
                                               `#hash((a . ,(second xs)))))
                      ((require racket/list) (define (tmpl xs)
                                               `#s(row ,(second xs))))))])
  (test-case (format "an unquote at depth one, or inside a box, hash table or prefab struct, is an access: ~s"
                     use)
    (define-values (ok? output) (gate-on use))
    (check-false ok? output)
    (check-regexp-match #rx"table-test[.]rkt:[0-9]+: second: read a test table's rows" output)))
