#lang racket/base

(require (only-in racket/file make-temporary-file)
         (only-in rackunit check-equal? test-case))

(define (call-with-module-file forms proc)
  (define file (make-temporary-file "review-~a.rkt"))
  (dynamic-wind void
                (lambda ()
                  (with-output-to-file file
                                       #:exists 'truncate
                                       (lambda ()
                                         (displayln "#lang racket/base")
                                         (for-each writeln forms)))
                  (proc file))
                (lambda () (delete-file file))))

(define (problems-in forms)
  (call-with-module-file forms
                         (lambda (file)
                           (parameterize ([current-namespace (make-base-namespace)])
                             (map (dynamic-require 'review/problem 'problem-message)
                                  ((dynamic-require 'review/lint 'lint) file))))))

(define quoted-data '((provide form) (define form '(define-job deposit x))))

(define quasiquoted-data '((provide form) (define x 1) (define form `(define-job deposit ,x))))

(define unquotes-in-compound-data
  '((provide forms) (define x 1)
                    (define y 2)
                    (define z 3)
                    (define forms (list `#& ,x `#hash((a . ,y)) `#s(p ,z)))))

(define test-forms-with-same-names
  '((require (only-in raft/tests/private/gpu test-gpu)) (test-gpu "a"
                                                          (define y 1)
                                                          y)
                                                        (test-gpu "b"
                                                          (define y 2)
                                                          y)))

(test-case "quoted data is not read as code"
  (check-equal? (problems-in quoted-data) '()))

(test-case "quasiquoted data is read only at its unquotes"
  (check-equal? (problems-in quasiquoted-data) '()))

(test-case "an unquote inside a quasiquoted box, hash value or prefab field is a use"
  (check-equal? (problems-in unquotes-in-compound-data) '()))

(test-case "each test form's body is a scope of its own"
  (check-equal? (problems-in test-forms-with-same-names) '()))
