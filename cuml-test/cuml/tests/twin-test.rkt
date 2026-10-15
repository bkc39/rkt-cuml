#lang racket/base

(require (only-in cuml/cluster/private/native libcuml-version)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in rackunit check-equal?)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/version_twin.py")

(test-twin "the Python twin imports the cuML release the cluster shim was built against"
  (define version (hash-ref (run-twin twin (hasheq)) 'version))
  (printf "twin version: cuml ~a, libcumlrkt_cluster ~a\n" version libcuml-version)
  (check-equal? version libcuml-version))
