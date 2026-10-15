#lang racket/base

(require (only-in cuml) ;; noqa
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in rackunit check-equal? check-false check-not-exn test-case)
         (only-in raft raft-version))

(define (fresh-namespace)
  (define ns (make-base-namespace))
  (for ([shared (in-list '(raft raft/unsafe))])
    (namespace-attach-module (current-namespace) shared ns))
  ns)

(test-case "cuml and cuml/cluster load, and the cluster shim accepts raft's ABI tag"
  (for ([module (in-list '(cuml cuml/cluster))])
    (check-not-exn (lambda ()
                     (parameterize ([current-namespace (fresh-namespace)])
                       (dynamic-require module #f))))))

(test-case "the cluster shim was built against the RAPIDS release of raft"
  (check-equal? libcuml-version (raft-version)))

(test-case "cuml/core loads no family's native library"
  (parameterize ([current-namespace (fresh-namespace)])
    (dynamic-require 'cuml/core #f)
    (check-false (module-declared? 'cuml/cluster/private/native #f))))
