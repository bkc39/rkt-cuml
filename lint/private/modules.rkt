#lang racket/base

(require (for-syntax racket/base
                     syntax/parse/pre)
         (only-in racket/file delete-directory/files make-directory* make-temporary-directory)
         (only-in syntax/modread with-module-reading-parameterization)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide call-with-temporary-directory
         read-module
         rkt?
         with-temporary-directory
         write-module)

(define (rkt? path)
  (define text (path->string path))
  (and (regexp-match? #rx"[.]rkt$" text) (not (regexp-match? #rx"/compiled/" text))))

(define (read-module file)
  (with-input-from-file file
                        (lambda ()
                          (port-count-lines! (current-input-port))
                          (with-module-reading-parameterization read-syntax))))

(define (write-module file language forms)
  (define-values (directory _name _dir?) (split-path file))
  (make-directory* directory)
  (with-output-to-file file
                       (lambda ()
                         (displayln language)
                         (for ([form (in-list forms)])
                           (writeln form)))))

(define (call-with-temporary-directory template proc)
  (define root (make-temporary-directory template))
  (dynamic-wind void
                (lambda () (proc root))
                (lambda () (delete-directory/files root #:must-exist? #f))))

(define-syntax-parse-rule (with-temporary-directory (root:id template:expr) body:expr ...+)
  (call-with-temporary-directory template
                                 (lambda (root)
                                   body ...)))
