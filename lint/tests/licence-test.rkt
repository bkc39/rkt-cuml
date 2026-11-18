#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-regexp-match test-case)
         (only-in "../licence.rkt" describe-reference gpl-references)
         (only-in "../private/modules.rkt" with-temporary-directory write-module))

(define (call-with-packages modules proc)
  (with-temporary-directory (root "licence-~a")
                            (for ([module (in-list modules)])
                              (match-define (list name language forms) module)
                              (write-module (build-path root name) language forms))
                            (proc root)))

(define (references modules)
  (call-with-packages
   modules
   (lambda (root)
     (for/list ([reference (in-list (gpl-references (build-path root "apache")
                                                    (build-path root "gpl")
                                                    '("gpl" "copyleft")))])
       (match-define (list file line name) reference)
       (list (regexp-replace #rx"^.*/apache/" (path->string file) "") line name)))))

(define shared-helper
  (list "gpl/cuml/tests/private/shared.rkt"
        "#lang racket/base"
        '((provide twice) (define (twice x)
                            (* 2 x)))))

(test-case "an Apache-2.0 module requiring a GPL module is a reference"
  (check-equal?
   (references (list shared-helper
                     (list "apache/cuml/tests/one-test.rkt"
                           "#lang racket/base"
                           '((require (only-in cuml/tests/private/shared twice)) (twice 1)))))
   '(("cuml/tests/one-test.rkt" 2 "cuml/tests/private/shared"))))

(test-case "a GPL module requiring an Apache-2.0 module is not"
  (check-equal?
   (references (list (list "apache/cuml/tests/private/shared.rkt"
                           "#lang racket/base"
                           '((provide twice) (define (twice x)
                                               (* 2 x))))
                     (list "gpl/cuml/tests/one-test.rkt"
                           "#lang racket/base"
                           '((require (only-in cuml/tests/private/shared twice)) (twice 1)))))
   '()))

(test-case "a dotted pair, and the GPL package's info.rkt, are not references"
  (check-equal?
   (references (list shared-helper
                     (list "gpl/info.rkt" "#lang info" '((define license 'GPL-2.0-or-later)))
                     (list "apache/cuml/tests/pairs-test.rkt"
                           "#lang racket/base"
                           '((define options '(width . 3)) (define level 'info)))))
   '()))

(test-case "an Apache-2.0 package depending on a GPL package is a reference"
  (check-equal? (references (list shared-helper
                                  (list "apache/info.rkt"
                                        "#lang info"
                                        '((define build-deps '("base" "copyleft"))))))
                '(("info.rkt" 2 "copyleft"))))

(define (one-reference path language forms)
  (references (list shared-helper (list path language forms))))

(test-case "a module of a GPL package's collection is a reference, by name or by lib"
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require (only-in copyleft/data rows))))
                '(("cuml/tests/a-test.rkt" 2 "copyleft/data")))
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require (lib "copyleft/plot.rkt"))))
                '(("cuml/tests/a-test.rkt" 2 "copyleft/plot")))
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((module+ test
                                   (require (prefix-in g: (lib "copyleft"))))))
                '(("cuml/tests/a-test.rkt" 2 "copyleft/main"))))

(test-case "a path that lands in the GPL package is a reference"
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require "../../../gpl/cuml/tests/private/shared.rkt")))
                '(("cuml/tests/a-test.rkt" 2 "../../../gpl/cuml/tests/private/shared.rkt")))
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require (file "../../../gpl/cuml/tests/private/shared.rkt"))))
                '(("cuml/tests/a-test.rkt" 2 "../../../gpl/cuml/tests/private/shared.rkt")))
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require (file "/elsewhere/copyleft/cuml/tests/shared.rkt"))))
                '(("cuml/tests/a-test.rkt" 2 "/elsewhere/copyleft/cuml/tests/shared.rkt")))
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((require "private/shared.rkt")))
                '()))

(test-case "a dynamic-require of a GPL module, by name or by collection-file-path, is a reference"
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((define (twice)
                                   (dynamic-require 'cuml/tests/private/shared 'twice))))
                '(("cuml/tests/a-test.rkt" 2 "cuml/tests/private/shared")))
  (check-equal?
   (one-reference "apache/cuml/tests/a-test.rkt"
                  "#lang racket/base"
                  '((dynamic-require (collection-file-path "shared.rkt" "cuml" "tests" "private")
                                     'twice)))
   '(("cuml/tests/a-test.rkt" 2 "cuml/tests/private/shared"))))

(test-case "a dependency on a GPL package by version or by URL is a reference"
  (check-equal?
   (one-reference "apache/info.rkt" "#lang info" '((define deps '(("copyleft" #:version "0.3")))))
   '(("info.rkt" 2 "copyleft")))
  (check-equal? (one-reference "apache/info.rkt"
                               "#lang info"
                               '((define build-deps '("git+https://github.com/someone/copyleft"))))
                '(("info.rkt" 2 "copyleft"))))

(test-case "a GPL package's name as data, outside a require, is not a reference"
  (check-equal? (one-reference "apache/cuml/tests/a-test.rkt"
                               "#lang racket/base"
                               '((define name "copyleft")
                                 (define solver 'copyleft)
                                 (define program '(require copyleft/data))
                                 (define path "../../../gpl/cuml/tests/private/shared.rkt")))
                '()))

(test-case "the message names the reference and where the helper goes"
  (check-regexp-match #rx"^one[.]rkt:3: gpl is GPL; .* lives in the Apache-2.0 package$"
                      (describe-reference (list "one.rkt" 3 "gpl"))))
