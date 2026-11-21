#lang racket/base

(require (only-in racket/file copy-directory/files make-parent-directory*)
         (only-in racket/match match-define)
         (only-in racket/port with-output-to-string)
         racket/runtime-path
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-false check-regexp-match check-true test-case)
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

(define (with-clean-tree add)
  (with-temporary-directory (root "no-twin-copies-~a")
                            (define tree (build-path root "tree"))
                            (copy-directory/files clean tree)
                            (add tree)
                            (run-gate tree)))

(test-case "a clean tree passes, its host, json.dump, sys.argv[2] and pins in their homes"
  (match-define (list passed? output) (with-clean-tree void))
  (check-true passed? output)
  (check-regexp-match #rx"^no-twin-copies: " output))

(test-case "a root without cuml-test fails and says so"
  (match-define (list passed? output)
    (with-temporary-directory (root "no-twin-copies-~a") (run-gate root)))
  (check-false passed? output)
  (check-regexp-match #rx"^no-twin-copies: no cuml-test under " output))

(test-case "a file grep cannot read fails the gate"
  (match-define (list passed? output)
    (with-clean-tree (lambda (tree)
                       (file-or-directory-permissions
                        (build-path tree "cuml-test/cuml/tests/python/clean_twin.py")
                        0))))
  (check-false passed? output)
  (check-regexp-match #rx"clean_twin.py: Permission denied" output))

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
