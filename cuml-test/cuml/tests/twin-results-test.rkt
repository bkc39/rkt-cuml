#lang racket/base

(require (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn exn:test:check? test-case)
         (only-in "private/python-env.rkt" checked-results python-cuml-version))

(define (answer version)
  (hasheq 'results '(1 2) 'version version))

(define pinned (hasheq 'cuml python-cuml-version 'sklearn "1.8.0"))

(define unpinned (hasheq 'cuml "26.10.00" 'sklearn "1.8.0"))

(define rows
  `(("pinned, no native" ,(answer pinned) #f ok)
    ("pinned, native agrees" ,(answer pinned) ,python-cuml-version ok)
    ("pinned, native differs" ,(answer pinned) "26.10.00" check)
    ("unpinned" ,(answer unpinned) #f check)
    ("unpinned, native agrees with it" ,(answer unpinned) "26.10.00" check)
    ("no cuml key" ,(answer (hasheq 'sklearn "1.8.0")) #f refused)
    ("a bare version string" ,(answer python-cuml-version) #f refused)
    ("no version" ,(hasheq 'results '(1 2)) #f refused)))

(for ([row (in-list rows)])
  (match-define (list name given expected outcome) row)
  (test-case name
    (define (go)
      (checked-results given #:expected-cuml expected))
    (match outcome
      ['ok (check-equal? (go) '(1 2))]
      ['check (check-exn exn:test:check? go)]
      ['refused (check-exn #rx"twin-results: the twin wrote no version.cuml" go)])))
