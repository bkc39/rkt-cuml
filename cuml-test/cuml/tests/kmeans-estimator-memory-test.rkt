#lang racket/base

(require (only-in cuml fit fit-predict fit-transform kmeans predict score transform)
         (only-in cuml/cluster/private/blobs blobs)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
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

(define rounds 100)
(define sample-every 50)
(define buffers-per-round 9)
(define n-samples 65536)
(define n-features 4)
(define n-clusters 8)

(define (data)
  (define-values (X _labels) (blobs n-samples n-features #:centers n-clusters #:seed 9))
  X)

(define (one-round e X)
  (define f (fit e X))
  (predict f X)
  (transform f X)
  (score f X)
  (fit-predict e X)
  (fit-transform e X))

(test-process-memory (format
                      "~a rounds of the estimator's fit, predict, transform, score, fit-predict and fit-transform free every buffer"
                      rounds)
  (define X (data))
  (define e (kmeans #:n-clusters n-clusters #:max-iter 20 #:random-state 1))
  (for ([_ (in-range 5)])
    (one-round e X))
  (define failures (settled-count release-failure-count))
  (match-define (list base-used base-bytes) (baseline-sample))
  (define live (jobs-live))
  (define threads (executor-threads))
  (define drops (settled-count buffer-drop-count))
  (define started (current-inexact-milliseconds))
  (define samples
    (for/list ([_ (in-range (quotient rounds sample-every))])
      (for ([_ (in-range sample-every)])
        (one-round e X))
      (pool-sample)))
  (define wanted (* buffers-per-round rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X e)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a estimator rounds in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed\n"
   rounds
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   base-used
   used
   base-bytes
   (map sample-bytes samples)
   freed
   wanted)
  (check-equal? freed wanted)
  (check-equal? (- (release-failure-count) failures) 0)
  (check-true (collect-until (lambda () (= (jobs-live) live))))
  (check-true (<= (executor-threads) (add1 threads)))
  (for ([u (in-list used)])
    (check-equal? u base-used "pool use returns to the baseline after a collection"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))
