#lang racket/base

(require (only-in racket/port with-output-to-string)
         (only-in racket/system system*)
         (only-in rackunit test-case)
         (for-syntax racket/base)
         syntax/parse/define)

(provide readable?
         run-gate
         test-unless-skipped)

(define (run-gate gate . arguments)
  (define ok? #f)
  (define output
    (with-output-to-string
     (lambda ()
       (parameterize ([current-error-port (current-output-port)])
         (set! ok? (apply system* (find-executable-path "bash") gate arguments))))))
  (list ok? output))

(define (readable? file)
  (with-handlers ([exn:fail:filesystem? (lambda (_) #f)])
    (call-with-input-file file void)
    #t))

(define-syntax-parse-rule (test-unless-skipped reason:expr name:expr body:expr ...+)
  (let ([why reason])
    (if why
        (printf "SKIP: ~a: ~a\n" why name)
        (test-case name
          body ...))))
