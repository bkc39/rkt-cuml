#lang racket/base

(require (only-in cuml) ;; noqa
         (only-in cuml/cluster kmeans-fit)
         (only-in cuml/cluster/private/native libcuml-version)
         (only-in cuml/preprocessing standard-scaler-fit)
         (only-in rackunit check-equal? check-false check-not-exn test-case)
         (only-in raft raft-version)
         (only-in "private/program.rkt" run-program))

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

(test-case "cuml re-exports cuml/cluster's k-means"
  (check-equal? (dynamic-require 'cuml 'kmeans-fit) kmeans-fit))

(test-case "cuml re-exports cuml/preprocessing's standard scaling"
  (check-equal? (dynamic-require 'cuml 'standard-scaler-fit) standard-scaler-fit))

(test-case "L20: cuml/preprocessing maps its own shim, and neither the cluster shim nor libcuml"
  (check-equal?
   (run-program `(module m racket/base
                   (require cuml/preprocessing
                            (only-in racket/file file->string))
                   (define maps (file->string "/proc/self/maps"))
                   (write (for/list ([library (in-list '("libcumlrkt_preprocessing.so"
                                                         "libcumlrkt_cluster.so"
                                                         "libcuml.so"))])
                            (regexp-match? (regexp-quote library) maps)))))
   '(0 "(#t #f #f)" "")))
