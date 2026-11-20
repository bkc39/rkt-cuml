#lang racket/base

(require (only-in cuml/cluster/private/native libcuml-version)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in "private/python-env.rkt" checked-results run-twin test-twin))

(define-runtime-path twin "python/version_twin.py")

(test-twin "the Python twin imports the cuML release the cluster shim was built against"
  (define answer (run-twin twin (hasheq 'cases '())))
  (checked-results answer #:expected-cuml libcuml-version)
  (printf "twin version: cuml ~a, libcumlrkt_cluster ~a\n"
          (hash-ref (hash-ref answer 'version) 'cuml)
          libcuml-version))
