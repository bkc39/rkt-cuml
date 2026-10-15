#lang racket/base

(require (only-in cuml exn:fail:cuml exn:fail:cuml?)
         (only-in rackunit check-equal? test-case)
         (only-in raft exn:fail:raft exn:fail:raft-kind exn:fail:raft?))

(test-case "the manual: a raft handler catches exn:fail:cuml and reads its kind"
  (check-equal? (with-handlers ([exn:fail:raft? exn:fail:raft-kind])
                  (raise (exn:fail:cuml "my-binding: n-clusters: expected at least 1, got 0"
                                        (current-continuation-marks)
                                        'logic)))
                'logic))

(test-case "the manual: exn:fail:cuml is both predicates' and keeps its kind"
  (define e (exn:fail:cuml "my-binding: out of memory" (current-continuation-marks) 'out-of-memory))
  (check-equal? (list (exn:fail:cuml? e) (exn:fail:raft? e) (exn:fail:raft-kind e))
                '(#t #t out-of-memory)))

(test-case "the manual: a raft failure is not a cuML failure"
  (check-equal? (exn:fail:cuml?
                 (exn:fail:raft "device-resources: no device 7" (current-continuation-marks) 'logic))
                #f))
