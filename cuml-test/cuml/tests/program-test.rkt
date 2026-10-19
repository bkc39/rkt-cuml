#lang racket/base

(require (only-in racket/match match)
         (only-in rackunit check-equal? fail test-case)
         (only-in "private/program.rkt" run-program))

(define (run-program/timeout program seconds)
  (define custodian (make-custodian))
  (define done (make-channel))
  (parameterize ([current-custodian custodian]
                 [current-subprocess-custodian-mode 'kill])
    (thread (lambda () (channel-put done (run-program program)))))
  (begin0 (sync/timeout seconds done)
    (custodian-shutdown-all custodian)))

(define stderr-bytes (* 256 1024))

(test-case "the runner returns when the child writes more than a pipe holds to stderr"
  (match (run-program/timeout `(module noisy racket/base
                                 (void (write-bytes (make-bytes ,stderr-bytes 120)
                                                    (current-error-port))))
                              20)
    [(list 0 "" errors) (check-equal? (string-length errors) stderr-bytes)]
    [#f (fail "the runner did not return within 20 s")]
    [(list status output errors)
     (fail (format "the child exited with ~a, ~a bytes out, ~a bytes err"
                   status
                   (string-length output)
                   (string-length errors)))]))
