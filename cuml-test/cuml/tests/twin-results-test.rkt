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
         (only-in "private/python-env.rkt"
                  checked-results
                  python-cuml-version
                  python-sklearn-version
                  twin-default-params
                  twin-results))

(define (answer version)
  (hasheq 'results '(1 2) 'version version))

(define pinned (hasheq 'cuml python-cuml-version 'sklearn python-sklearn-version))

(define unpinned (hasheq 'cuml "26.10.00" 'sklearn python-sklearn-version))

(define stale-sklearn (hasheq 'cuml python-cuml-version 'sklearn "1.7.0"))

(define cuml-only (hasheq 'cuml python-cuml-version))

(define pin "the twin's Python cuML is not the pinned release")

(define native "the twin's Python cuML is not the native libcuml")

(define sklearn-pin "the twin's scikit-learn is not the pinned release")

(define (failed-message thunk)
  (with-handlers ([exn:test:check? (lambda (e)
                                     (for/first ([info (in-list (exn:test:check-stack e))]
                                                 #:when (eq? (check-info-name info) 'message))
                                       (check-info-value info)))])
    (thunk)
    'passed))

(define rows
  `(("pinned, no native" ,(answer pinned) #f #f ok)
    ("pinned, native agrees" ,(answer pinned) ,python-cuml-version #f ok)
    ("pinned, native differs" ,(answer pinned) "26.10.00" #f (check ,native))
    ("unpinned" ,(answer unpinned) #f #f (check ,pin))
    ("unpinned, native agrees with it" ,(answer unpinned) "26.10.00" #f (check ,pin))
    ("sklearn pinned and expected" ,(answer pinned) #f ,python-sklearn-version ok)
    ("sklearn differs from the expected" ,(answer stale-sklearn)
                                         #f
                                         ,python-sklearn-version
                                         (check ,sklearn-pin))
    ("no sklearn key, sklearn expected" ,(answer cuml-only)
                                        #f
                                        ,python-sklearn-version
                                        (check ,sklearn-pin))
    ("no sklearn key, none expected" ,(answer cuml-only) #f #f ok)
    ("a stale sklearn, none expected" ,(answer stale-sklearn) #f #f ok)
    ("no cuml key" ,(answer (hasheq 'sklearn python-sklearn-version))
                   #f
                   #f
                   (refused "no version.cuml"))
    ("a bare version string" ,(answer python-cuml-version) #f #f (refused "no version.cuml"))
    ("no version" ,(hasheq 'results '(1 2)) #f #f (refused "no version.cuml"))
    ("no results" ,(hasheq 'version pinned) #f #f (refused "no results"))
    ("not a hash" ,'(1 2) #f #f (refused "no results"))))

(for ([row (in-list rows)])
  (match-define (list name given expected expected-sklearn outcome) row)
  (test-case name
    (define (go)
      (checked-results given #:expected-cuml expected #:expected-sklearn expected-sklearn))
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
                                                 (expected-sklearn . #f)
                                                 (request . ,(hasheq 'cases requests))
                                                 (outcome . passed))
    ((name . "the fixtures reach the request") (version . ,pinned)
                                               (fixtures . ,fixtures)
                                               (expected . #f)
                                               (expected-sklearn . #f)
                                               (request . ,(hasheq 'cases requests
                                                                   'fixtures fixtures))
                                               (outcome . passed))
    ((name . "the native libcuml given is checked") (version . ,pinned)
                                                    (fixtures . #f)
                                                    (expected . "26.10.00")
                                                    (expected-sklearn . #f)
                                                    (request . ,(hasheq 'cases requests))
                                                    (outcome . ,native))
    ((name . "an unpinned twin fails without a native libcuml") (version . ,unpinned)
                                                                (fixtures . #f)
                                                                (expected . #f)
                                                                (expected-sklearn . #f)
                                                                (request . ,(hasheq 'cases requests))
                                                                (outcome . ,pin))
    ((name . "the expected sklearn is checked") (version . ,stale-sklearn)
                                                (fixtures . #f)
                                                (expected . #f)
                                                (expected-sklearn . ,python-sklearn-version)
                                                (request . ,(hasheq 'cases requests))
                                                (outcome . ,sklearn-pin))))

(for ([row (in-list run-rows)])
  (match-define (hash-table ('name name)
                            ('version version)
                            ('fixtures sent-fixtures)
                            ('expected expected)
                            ('expected-sklearn expected-sklearn)
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
                                                  #:expected-sklearn expected-sklearn
                                                  #:fixtures sent-fixtures
                                                  #:run run)))
                  outcome)
    (check-equal? sent (list "twin.py" request))))

(define scaler-params (hasheq 'copy #t 'with_mean #t 'with_std #f))

(define params-rows
  `(("the scaler's params check passes its pinned sklearn" ,pinned
                                                           passed
                                                           ,(hasheq 'with-mean #t 'with-std #f))
    ("the scaler's params check refuses a stale sklearn" ,stale-sklearn ,sklearn-pin #f)))

(for ([row (in-list params-rows)])
  (match-define (list name version outcome expected-params) row)
  (test-case name
    (define params #f)
    (define (run script input)
      (hasheq 'results (list scaler-params) 'version version))
    (check-equal? (failed-message (lambda ()
                                    (set! params
                                          (twin-default-params "standard_scaler_twin.py"
                                                               #:unbound '(copy)
                                                               #:expected-sklearn
                                                               python-sklearn-version
                                                               #:run run))))
                  outcome)
    (check-equal? params expected-params)))
