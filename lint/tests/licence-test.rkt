#lang racket/base

(require (only-in racket/file delete-directory/files make-directory* make-temporary-directory)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-regexp-match test-case)
         (only-in "../licence.rkt" describe-reference gpl-references))

(define (call-with-packages modules proc)
  (define root (make-temporary-directory "licence-~a"))
  (dynamic-wind void
                (lambda ()
                  (for ([module (in-list modules)])
                    (match-define (list name language forms) module)
                    (define file (build-path root name))
                    (make-directory* (let-values ([(directory _name _dir?) (split-path file)])
                                       directory))
                    (with-output-to-file file
                                         (lambda ()
                                           (displayln language)
                                           (for-each writeln forms))))
                  (proc root))
                (lambda () (delete-directory/files root))))

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

(test-case "the message names the reference and where the helper goes"
  (check-regexp-match #rx"^one[.]rkt:3: gpl is GPL; .* lives in the Apache-2.0 package$"
                      (describe-reference (list "one.rkt" 3 "gpl"))))
