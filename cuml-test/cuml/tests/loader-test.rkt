#lang racket/base

(require (only-in cuml/core exn:fail:cuml?)
         (only-in cuml/core/private/install-native not-staged-advice)
         (only-in cuml/core/private/native load-family)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-false check-pred check-regexp-match test-case)
         (only-in raft exn:fail:raft-kind)
         (only-in raft/private/resource with-release))

(define shared-object-without-entries
  (call-with-input-file "/proc/self/maps"
                        (lambda (in)
                          (match (regexp-match #px"/\\S+/libm[.]so[.]6" in)
                            [(list path) (bytes->path path)]))))

(define (load-failure stage!)
  (with-release ([dir
                  (make-temporary-directory "cuml-loader-~a")
                  (lambda (dir) (delete-directory/files dir #:must-exist? #f))])
    (stage! (build-path dir "libcumlrkt_cluster.so"))
    (with-handlers ([exn:fail? values])
      (load-family "cluster" dir)
      #f)))

(define (check-load-failure e problem)
  (check-pred exn:fail:cuml? e)
  (check-equal? (exn:fail:raft-kind e) 'generic)
  (check-regexp-match problem (exn-message e))
  (check-regexp-match (regexp-quote not-staged-advice) (exn-message e)))

(test-case "a family whose library is not staged raises exn:fail:cuml with the advice"
  (check-load-failure (load-failure void) #rx"^cuml/cluster: libcumlrkt_cluster is not staged\n"))

(test-case "a staged file that is no shared object raises exn:fail:cuml with the advice"
  (check-load-failure
   (load-failure (lambda (path)
                   (call-with-output-file path (lambda (out) (write-bytes #"not a library" out)))))
   #rx"^cuml/cluster: libcumlrkt_cluster is staged but would not load\n"))

(test-case "a stale library without the family's entries raises exn:fail:cuml with the advice"
  (define e (load-failure (lambda (path) (copy-file shared-object-without-entries path))))
  (check-load-failure
   e
   #rx"^cuml/cluster: libcumlrkt_cluster lacks entry cuml_cluster_last_error; it is stale, restage it\n")
  (check-false (regexp-match? #rx"driver|LD_LIBRARY_PATH" (exn-message e))
               "a missing entry is a stale library, not a driver problem"))
