#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/port port->string)
         (only-in raft/private/resource with-release))

(provide run-program)

(define (run-program program)
  (with-release ([dir
                  (make-temporary-directory "cuml-program-~a")
                  (lambda (dir) (delete-directory/files dir #:must-exist? #f))])
    (define path (build-path dir "program.rkt"))
    (call-with-output-file path (lambda (out) (write program out)))
    (define-values (process stdout stdin stderr) (subprocess #f #f #f (find-exe) path))
    (close-output-port stdin)
    (define errors (make-channel))
    (thread (lambda () (channel-put errors (port->string stderr))))
    (define output (port->string stdout))
    (define error-output (channel-get errors))
    (subprocess-wait process)
    (list (subprocess-status process) output error-output)))
