#lang racket/base

(require (only-in cuml/cluster/private/blobs blobs blobs/short-call)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until drain-finalizers! settled-count)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/device-memory.rkt" device-free-bytes))

(define calls 500)
(define sample-every 100)
(define buffers-per-call 2)
(define device-slack (* 256 1024 1024))

(define (pool-sample)
  (drain-finalizers!)
  (list (probe-pool-used 0) (device-free-bytes)))

(define (check-device-free baseline-free lowest-free)
  (define floor-free (- baseline-free device-slack))
  (define settled-free
    (if (>= lowest-free floor-free)
        lowest-free
        (cadr (pool-sample))))
  (check-true (>= settled-free floor-free)
              (format "device-wide free memory fell ~a bytes below the baseline (twice)"
                      (- baseline-free settled-free))))

(test-pools "500 blobs calls, jobs and short calls alike, leave device memory flat and free every buffer"
  (for ([make-blobs (in-list (list blobs blobs/short-call))])
    (for ([_ (in-range 10)])
      (make-blobs 256 4 #:centers 3 #:seed 5))
    (define failures (settled-count release-failure-count))
    (match-define (list base-used base-free) (pool-sample))
    (define live (jobs-live))
    (define threads (executor-threads))
    (define drops (settled-count buffer-drop-count))
    (define samples
      (for/list ([_ (in-range (quotient calls sample-every))])
        (for ([_ (in-range sample-every)])
          (make-blobs 256 4 #:centers 3 #:seed 5))
        (pool-sample)))
    (define wanted (* buffers-per-call calls))
    (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
    (define freed (- (buffer-drop-count) drops))
    (define used (map car samples))
    (define lowest-free (apply min (map cadr samples)))
    (printf
     "memory: ~a ~a calls; pool used ~a bytes before, ~a after each ~a; device free ~a before, lowest ~a after (cuMemGetInfo, shared GPU); ~a buffers freed of ~a allocated; ~a release failures\n"
     calls
     (object-name make-blobs)
     base-used
     used
     sample-every
     base-free
     lowest-free
     freed
     wanted
     (- (release-failure-count) failures))
    (check-true (>= base-used 0) "the probe reads raft's pool")
    (check-equal? freed wanted)
    (check-equal? (- (release-failure-count) failures) 0)
    (check-true (collect-until (lambda () (= (jobs-live) live))))
    (check-true (<= (executor-threads) (add1 threads)))
    (for ([u (in-list used)])
      (check-equal? u base-used "pool use returns to the baseline after a collection"))
    (check-device-free base-free lowest-free)))
