#lang racket/base

(require (only-in racket/match match match-define)
         (only-in rackunit
                  check-equal?
                  check-exn
                  check-info-name
                  check-info-value
                  exn:test:check-stack
                  exn:test:check?
                  test-case)
         (only-in "private/python-env.rkt" checked-results python-cuml-version twin-results))

(define (answer version)
  (hasheq 'results '(1 2) 'version version))

(define pinned (hasheq 'cuml python-cuml-version 'sklearn "1.8.0"))

(define unpinned (hasheq 'cuml "26.10.00" 'sklearn "1.8.0"))

(define pin "the twin's Python cuML is not the pinned release")

(define native "the twin's Python cuML is not the native libcuml")

(define (failed-message thunk)
  (with-handlers ([exn:test:check? (lambda (e)
                                     (for/first ([info (in-list (exn:test:check-stack e))]
                                                 #:when (eq? (check-info-name info) 'message))
                                       (check-info-value info)))])
    (thunk)
    'passed))

(define rows
  `(("pinned, no native" ,(answer pinned) #f ok)
    ("pinned, native agrees" ,(answer pinned) ,python-cuml-version ok)
    ("pinned, native differs" ,(answer pinned) "26.10.00" (check ,native))
    ("unpinned" ,(answer unpinned) #f (check ,pin))
    ("unpinned, native agrees with it" ,(answer unpinned) "26.10.00" (check ,pin))
    ("no cuml key" ,(answer (hasheq 'sklearn "1.8.0")) #f (refused "no version.cuml"))
    ("a bare version string" ,(answer python-cuml-version) #f (refused "no version.cuml"))
    ("no version" ,(hasheq 'results '(1 2)) #f (refused "no version.cuml"))
    ("no results" ,(hasheq 'version pinned) #f (refused "no results"))
    ("not a hash" ,'(1 2) #f (refused "no results"))))

(for ([row (in-list rows)])
  (match-define (list name given expected outcome) row)
  (test-case name
    (define (go)
      (checked-results given #:expected-cuml expected))
    (match outcome
      ['ok (check-equal? (go) '(1 2))]
      [(list 'check message) (check-equal? (failed-message go) message)]
      [(list 'refused what)
       (check-exn (regexp (format "^twin-results: the twin wrote ~a" what)) go)])))

(define requests (list (hasheq 'op "params")))

(define fixtures (hasheq 'a '(1.0)))

(define run-rows
  `(((name . "no fixtures sends only the cases") (version . ,pinned)
                                                 (fixtures . #f)
                                                 (expected . #f)
                                                 (request . ,(hasheq 'cases requests))
                                                 (outcome . passed))
    ((name . "the fixtures reach the request") (version . ,pinned)
                                               (fixtures . ,fixtures)
                                               (expected . #f)
                                               (request . ,(hasheq 'cases requests
                                                                   'fixtures fixtures))
                                               (outcome . passed))
    ((name . "the native libcuml given is checked") (version . ,pinned)
                                                    (fixtures . #f)
                                                    (expected . "26.10.00")
                                                    (request . ,(hasheq 'cases requests))
                                                    (outcome . ,native))
    ((name . "an unpinned twin fails without a native libcuml") (version . ,unpinned)
                                                                (fixtures . #f)
                                                                (expected . #f)
                                                                (request . ,(hasheq 'cases requests))
                                                                (outcome . ,pin))))

(for ([row (in-list run-rows)])
  (match-define (hash-table ('name name)
                            ('version version)
                            ('fixtures sent-fixtures)
                            ('expected expected)
                            ('request request)
                            ('outcome outcome))
    (make-immutable-hasheq row))
  (test-case name
    (define sent '())
    (define (run script input)
      (set! sent (list script input))
      (answer version))
    (check-equal? (failed-message (lambda ()
                                    (twin-results "twin.py"
                                                  requests
                                                  #:expected-cuml expected
                                                  #:fixtures sent-fixtures
                                                  #:run run)))
                  outcome)
    (check-equal? sent (list "twin.py" request))))
