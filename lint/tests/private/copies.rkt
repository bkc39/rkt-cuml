#lang racket/base

(require (only-in racket/match match-define)
         (only-in "../../duplicates.rkt" copied-definitions-in describe-copies)
         (only-in "../../private/modules.rkt" with-temporary-directory write-module))

(provide call-with-modules
         copies)

(define (call-with-modules modules proc)
  (with-temporary-directory (root "duplicates-~a")
                            (for ([module (in-list modules)])
                              (match-define (list name forms) module)
                              (write-module (build-path root name) "#lang racket/base" forms))
                            (proc root)))

(define (copies modules)
  (call-with-modules modules
                     (lambda (root)
                       (for/list ([group (in-list (copied-definitions-in (list root)))])
                         (for/list ([line (in-list (cdr (regexp-split #rx"\n  "
                                                                      (describe-copies group))))])
                           (regexp-replace #rx"^.*/" line ""))))))
