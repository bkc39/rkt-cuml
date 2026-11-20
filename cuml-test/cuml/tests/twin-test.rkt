#lang racket/base

(require (only-in cuml/cluster/private/native libcuml-version)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" python-cuml-version test-twin twin-results))

(define-runtime-path twin "python/version_twin.py")

(test-twin "the Python twin imports the cuML release the cluster shim was built against"
  (printf "twin version: cuml ~a, libcumlrkt_cluster ~a\n" python-cuml-version libcuml-version)
  (check-equal? (twin-results twin '() #:expected-cuml libcuml-version) '()))
