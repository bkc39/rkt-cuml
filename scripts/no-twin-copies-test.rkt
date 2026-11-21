#lang racket/base

(require (only-in racket/file copy-directory/files make-parent-directory*)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-false check-regexp-match check-true test-case)
         (only-in "../lint/private/modules.rkt" with-temporary-directory)
         (only-in "private/gate-run.rkt" readable? run-gate test-unless-skipped))

(define-runtime-path gate "no-twin-copies.sh")

(define-runtime-path clean "fixtures/no-twin-copies/clean")

(define-runtime-path violations "fixtures/no-twin-copies/violations")

(define (gate-on root)
  (run-gate gate (path->string root)))

(define (call-with-clean-tree proc)
  (with-temporary-directory (root "no-twin-copies-~a")
                            (define tree (build-path root "tree"))
                            (copy-directory/files clean tree)
                            (proc tree)))

(define (with-clean-tree add)
  (call-with-clean-tree (lambda (tree)
                          (add tree)
                          (gate-on tree))))

(test-case "a clean tree passes, its host, json.dump, sys.argv[2] and pins in their homes"
  (match-define (list passed? output) (with-clean-tree void))
  (check-true passed? output)
  (check-regexp-match #rx"^no-twin-copies: " output))

(test-case "a root without cuml-test fails and says so"
  (match-define (list passed? output)
    (with-temporary-directory (root "no-twin-copies-~a") (gate-on root)))
  (check-false passed? output)
  (check-regexp-match #rx"^no-twin-copies: no cuml-test under " output))

(call-with-clean-tree (lambda (tree)
                        (define file (build-path tree "cuml-test/cuml/tests/python/clean_twin.py"))
                        (file-or-directory-permissions file 0)
                        (test-unless-skipped (and (readable? file)
                                                  "the unreadable file reads (root?)")
                                             "a file grep cannot read fails the gate"
                          (match-define (list passed? output) (gate-on tree))
                          (check-false passed? output)
                          (check-regexp-match #rx"clean_twin.py: Permission denied" output))))

(define rows
  '(("def host( outside twin_common.py fails"
     "host_twin.py"
     "cuml-test/cuml/tests/python/host_twin.py"
     #rx"python/host_twin.py:4: def host\\( outside [^\n]*: import host from twin_common")
    ("json.dump( outside twin_common.py fails"
     "dump_twin.py"
     "cuml-test/cuml/tests/python/dump_twin.py"
     #rx"python/dump_twin.py:7: json.dump\\( outside [^\n]*: write the twin's output with")
    ("sys.argv[2] outside twin_common.py fails"
     "argv_twin.py"
     "cuml-test/cuml/tests/python/argv_twin.py"
     #rx"python/argv_twin.py:6: sys.argv\\[2\\] outside [^\n]*: write the twin's output with")
    ("a quoted cuML pin outside python-env.rkt fails"
     "cuml-pin.rkt"
     "cuml-test/cuml/tests/cuml-pin-test.rkt"
     #rx"tests/cuml-pin-test.rkt:5: a pinned Python version outside [^\n]*python-env.rkt")
    ("a quoted sklearn pin outside python-env.rkt fails"
     "sklearn-pin.rkt"
     "cuml-test/cuml/tests/private/scaler-data.rkt"
     #rx"private/scaler-data.rkt:5: a pinned Python version outside [^\n]*python-env.rkt")
    ("a quoted cuML pin in a Python twin fails"
     "pin_twin.py"
     "cuml-test/cuml/tests/python/pin_twin.py"
     #rx"python/pin_twin.py:5: a pinned Python version outside [^\n]*python-env.rkt")))

(for ([row (in-list rows)])
  (match-define (list name violation target message) row)
  (test-case name
    (match-define (list passed? output)
      (with-clean-tree (lambda (tree)
                         (define file (build-path tree target))
                         (make-parent-directory* file)
                         (copy-file (build-path violations violation) file))))
    (check-false passed? output)
    (check-regexp-match message output)))
