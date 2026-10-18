#lang racket/base

(require (only-in cuml/cluster/private/blobs blobs blobs/short-call)
         (only-in racket/list last)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory))

(define calls 500)
(define sample-every 100)
(define buffers-per-call 2)
(define n-samples 65536)
(define n-features 4)

(define (call-blobs make-blobs)
  (call-with-values (lambda () (make-blobs n-samples n-features #:centers 3 #:seed 5)) list))

(define (warm-up make-blobs)
  (for ([_ (in-range 10)])
    (call-blobs make-blobs)))

(define (sampled-calls make-blobs keep!)
  (for/list ([_ (in-range (quotient calls sample-every))])
    (for ([_ (in-range sample-every)])
      (keep! (call-blobs make-blobs)))
    (pool-sample)))

(test-process-memory "500 blobs calls, jobs and short calls alike, leave device memory flat and free every buffer"
  (for ([make-blobs (in-list (list blobs blobs/short-call))])
    (warm-up make-blobs)
    (define failures (settled-count release-failure-count))
    (match-define (list base-used base-bytes) (baseline-sample))
    (define live (jobs-live))
    (define threads (executor-threads))
    (define drops (settled-count buffer-drop-count))
    (define samples (sampled-calls make-blobs void))
    (define wanted (* buffers-per-call calls))
    (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
    (define freed (- (buffer-drop-count) drops))
    (define used (map sample-used samples))
    (define growth (device-growth base-bytes samples))
    (printf
     "memory: ~a ~a calls of ~ax~a; pool used ~a bytes before, ~a after each ~a; this process's device memory ~a bytes before, ~a after each ~a; ~a buffers freed of ~a allocated; ~a release failures\n"
     calls
     (object-name make-blobs)
     n-samples
     n-features
     base-used
     used
     sample-every
     base-bytes
     (map sample-bytes samples)
     sample-every
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
    (check-true
     (<= growth device-slack)
     (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack))))

(test-process-memory "the device-memory check fails when every call's arrays stay alive"
  (warm-up blobs)
  (match-define (list _ base-bytes) (baseline-sample))
  (define kept '())
  (define samples (sampled-calls blobs (lambda (arrays) (set! kept (cons arrays kept)))))
  (match-define (list _ held-bytes) (last samples))
  (printf
   "memory: ~a blobs calls kept alive; this process's device memory ~a bytes before, ~a after\n"
   calls
   base-bytes
   held-bytes)
  (check-true (> (- held-bytes base-bytes) device-slack) "the kept arrays exceed the device slack"))
