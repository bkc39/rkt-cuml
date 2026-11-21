#lang racket/base

(require (only-in rackunit
                  check-equal?
                  check-info-name
                  check-info-value
                  current-check-around
                  exn:test:check-stack
                  exn:test:check?
                  test-case)
         (only-in "private/linear-data.rkt" check-law))

(define (failure-messages thunk)
  (define failures '())
  (define (keep! e)
    (set! failures (cons e failures)))
  (parameterize ([current-check-around (lambda (check)
                                         (with-handlers ([exn:test:check? keep!])
                                           (check)))])
    (thunk))
  (for*/list ([e (in-list (reverse failures))]
              [info (in-list (exn:test:check-stack e))]
              #:when (eq? (check-info-name info) 'message))
    (check-info-value info)))

(test-case "check-law fails a non-finite bound and names the law"
  (check-equal? (failure-messages (lambda () (check-law "x" 0 +inf.0))) '("x: the bound is +inf.0")))
