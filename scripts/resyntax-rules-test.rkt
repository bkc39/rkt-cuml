#lang racket/base

(module+ test
  (require (only-in racket/port call-with-output-string)
           (only-in rackunit check-equal? test-case)
           resyntax/test/private/rackunit
           (only-in syntax/parse/pre syntax-parse)
           "resyntax-rules.rkt")

  (define (relayed formals)
    (syntax-parse (datum->syntax #f formals)
      [(formal:relay-formal ...) #t]
      [_ #f]))

  (test-case "L-4: a relay formal is a positional id or an optional keyword argument"
    (check-equal? (map relayed '((x) (#:k [k 1]) (x #:k [k 1]) ([y 2]) (#:k k) (#:k)))
                  '(#t #t #t #f #f #f)))

  (define (program . forms)
    (code-block (call-with-output-string
                 (lambda (out)
                   (displayln "#lang racket/base" out)
                   (writeln '(require (only-in cuml/cluster/private/define-who define/who)) out)
                   (for ([form (in-list forms)])
                     (writeln form out))))))

  (define relay
    '(define (twice/who who x #:times [n 2])
       (list who x n)))

  (define wrapper
    '(define (twice x #:times [n 2])
       (twice/who 'twice x #:times n)))

  (parameterize ([current-suite-under-test cuml-rules])
    (test-case "a procedure and its /who relay become a define/who"
      (check-suite-refactors (program `(define (run)
                                         ,relay
                                         ,wrapper
                                         (twice 1)))
                             (program '(define (run)
                                         (define/who (twice who x #:times [n 2]) (list who x n))
                                         (twice 1)))))

    (test-case "a private relay without a wrapper stays"
      (check-suite-does-not-refactor (program '(define (run)
                                                 (define (helper who x)
                                                   (list who x))
                                                 (define (twice x)
                                                   (helper 'twice x))
                                                 (twice 1)))))

    (test-case "a wrapper that passes other arguments stays"
      (check-suite-does-not-refactor (program `(define (run)
                                                 ,relay
                                                 (define (twice x #:times [n 2])
                                                   (twice/who 'twice x #:times 3))
                                                 (twice 1)))))

    (test-case "a relay with an optional positional argument stays"
      (check-suite-does-not-refactor (program '(define (run)
                                                 (define (twice/who who x [y 2])
                                                   (list who x y))
                                                 (define (twice x [y 2])
                                                   (twice/who 'twice x y))
                                                 (twice 1)))))

    (test-case "a relay with a required keyword argument stays"
      (check-suite-does-not-refactor (program '(define (run)
                                                 (define (twice/who who x #:times n)
                                                   (list who x n))
                                                 (define (twice x #:times n)
                                                   (twice/who 'twice x #:times n))
                                                 (twice 1 #:times 2)))))))
