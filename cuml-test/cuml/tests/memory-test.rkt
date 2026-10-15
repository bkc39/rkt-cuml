#lang racket/base

(require (only-in cuml/cluster/private/blobs blobs)
         (only-in rackunit check-equal? check-true)
         (only-in raft shape)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job jobs-live)
         (only-in raft/tests/private/collect collect-until drain-finalizers! settled-count)
         (only-in raft/tests/private/gpu test-gpu))

(define calls 500)
(define buffers-per-call 2)

(define (blobs-once job?)
  (define-values (X labels) (blobs 256 4 #:centers 3 #:seed 5 #:job? job?))
  (map shape (list X labels)))

(test-gpu "500 blobs calls, jobs and short calls alike, free every buffer they allocate"
  (for ([job? (in-list '(#t #f))])
    (for ([_ (in-range 10)])
      (blobs-once job?))
    (drain-finalizers!)
    (define failures (settled-count release-failure-count))
    (define live (jobs-live))
    (define drops (settled-count buffer-drop-count))
    (for ([_ (in-range calls)])
      (blobs-once job?))
    (define wanted (* buffers-per-call calls))
    (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
    (define freed (- (buffer-drop-count) drops))
    (printf "memory: ~a ~a calls; ~a buffers freed of ~a allocated; ~a release failures\n"
            calls
            (if job? "job" "short")
            freed
            wanted
            (- (release-failure-count) failures))
    (check-equal? freed wanted)
    (check-equal? (- (release-failure-count) failures) 0)
    (check-true (collect-until (lambda () (= (jobs-live) live))))))
