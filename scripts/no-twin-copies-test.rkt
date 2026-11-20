#lang racket/base

(require (only-in racket/file copy-directory/files make-parent-directory*)
         (only-in racket/match match-define)
         (only-in racket/port with-output-to-string)
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-regexp-match test-case)
         (only-in "../lint/private/modules.rkt" with-temporary-directory))

(define-runtime-path gate "no-twin-copies.sh")

(define-runtime-path clean "fixtures/no-twin-copies/clean")

(define-runtime-path violations "fixtures/no-twin-copies/violations")

(define (run-gate root)
  (define ok? #f)
  (define output
    (with-output-to-string
     (lambda ()
       (parameterize ([current-error-port (current-output-port)])
         (set! ok? (system* (find-executable-path "bash") gate (path->string root)))))))
  (list ok? output))

(define rows
  '(("a clean tree passes, its host, json.dump and pins in their homes" #f
                                                                        #f
                                                                        #t
                                                                        #rx"^no-twin-copies: ")
    ("def host( outside twin_common.py fails"
     "host_twin.py"
     "cuml-test/cuml/tests/python/host_twin.py"
     #f
     #rx"python/host_twin.py:4: def host\\( outside [^\n]*: import host from twin_common")
    ("json.dump( outside twin_common.py fails"
     "dump_twin.py"
     "cuml-test/cuml/tests/python/dump_twin.py"
     #f
     #rx"python/dump_twin.py:7: json.dump\\( outside [^\n]*: write the twin's output with")
    ("a quoted cuML pin outside python-env.rkt fails"
     "cuml-pin.rkt"
     "cuml-test/cuml/tests/cuml-pin-test.rkt"
     #f
     #rx"tests/cuml-pin-test.rkt:5: a pinned Python version outside [^\n]*python-env.rkt")
    ("a quoted sklearn pin outside python-env.rkt fails"
     "sklearn-pin.rkt"
     "cuml-test/cuml/tests/private/scaler-data.rkt"
     #f
     #rx"private/scaler-data.rkt:5: a pinned Python version outside [^\n]*python-env.rkt")))

(for ([row (in-list rows)])
  (match-define (list name violation target ok? message) row)
  (test-case name
    (with-temporary-directory (root "no-twin-copies-~a")
                              (define tree (build-path root "tree"))
                              (copy-directory/files clean tree)
                              (when violation
                                (define file (build-path tree target))
                                (make-parent-directory* file)
                                (copy-file (build-path violations violation) file))
                              (match-define (list passed? output) (run-gate tree))
                              (check-equal? passed? ok? output)
                              (check-regexp-match message output))))
