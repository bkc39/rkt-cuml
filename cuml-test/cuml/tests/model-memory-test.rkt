#lang racket/base

(require (only-in cuml/core/private/model native-model-release! with-native-models)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/foreign/memory release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until drain-finalizers! settled-count)
         (only-in raft/tests/private/probe probe-pool-used)
         (only-in "private/device-memory.rkt" process-device-bytes)
         (only-in "private/model-probe.rkt"
                  check-model-probe
                  counts
                  counts-created
                  counts-destroyed
                  counts-released
                  counts-stale
                  probe-counts
                  probe-model)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory))

(define MiB (* 1024 1024))
(define rounds 200)
(define sample-every 50)
(define models-per-round 4)
(define releases-per-round 3)
(define payload (* 8 MiB))
(define region-length 1024)

(define (model)
  (probe-model #:payload-bytes payload #:region-length region-length))

(define (one-round)
  (let-values ([(_m _region) (model)])
    (void))
  (let-values ([(m _region) (model)])
    (native-model-release! m))
  (let-values ([(m _region) (model)])
    (with-native-models ([held m]) (void held)))
  (define-values (m kept) (model))
  (native-model-release! m)
  kept)

(define (counts- after before)
  (match-define (counts c r d s) after)
  (match-define (counts c0 r0 d0 s0) before)
  (counts (- c c0) (- r r0) (- d d0) (- s s0)))

(test-process-memory (format
                      "M1: ~a rounds of ~a models, dropped or released, leave device memory flat"
                      rounds
                      models-per-round)
  (check-model-probe)
  (for ([_ (in-range 5)])
    (one-round))
  (define failures (settled-count release-failure-count))
  (match-define (list base-used base-bytes) (baseline-sample))
  (define live (jobs-live))
  (define threads (executor-threads))
  (define before (probe-counts))
  (define started (current-inexact-milliseconds))
  (define samples
    (for/list ([_ (in-range (quotient rounds sample-every))])
      (define kept
        (for/list ([_ (in-range sample-every)])
          (one-round)))
      (void/reference-sink kept)
      (pool-sample)))
  (check-true (collect-until (lambda ()
                               (define now (counts- (probe-counts) before))
                               (= (counts-destroyed now) (counts-created now)))))
  (define delta (counts- (probe-counts) before))
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a model rounds in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a created, ~a released, ~a destroyed, ~a stale\n"
   rounds
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   base-used
   used
   base-bytes
   (map sample-bytes samples)
   (counts-created delta)
   (counts-released delta)
   (counts-destroyed delta)
   (counts-stale delta))
  (check-equal? (counts-created delta) (* models-per-round rounds))
  (check-equal? (counts-destroyed delta) (* models-per-round rounds))
  (check-equal? (counts-released delta) (* releases-per-round rounds))
  (check-equal? (counts-stale delta) 0)
  (check-equal? (- (release-failure-count) failures) 0)
  (check-true (collect-until (lambda () (= (jobs-live) live))))
  (check-true (<= (executor-threads) (add1 threads)))
  (for ([u (in-list used)])
    (check-equal? u base-used "pool use returns to the baseline after a collection"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))

(test-process-memory "M2: a release frees the payload at once, before any collection"
  (check-model-probe)
  (define-values (m region) (model))
  (define used (probe-pool-used device))
  (native-model-release! m)
  (check-equal? (- used (probe-pool-used device)) payload)
  (void/reference-sink region))

(define (peak-growth-creating n big)
  (define start (process-device-bytes #:device device))
  (for/fold ([peak 0]) ([_ (in-range n)])
    (let-values ([(_m _region) (probe-model #:payload-bytes big #:region-length 1)])
      (void))
    (max peak (- (process-device-bytes #:device device) start))))

(test-process-memory "M3: unreleased models are collected under GC pressure alone"
  (check-model-probe)
  (define big (* 64 MiB))
  (drain-finalizers!)
  (define live (current-memory-use))
  (define bound (+ (* 2 big) (* 8192 (sqrt live)) device-slack))
  (define peak (peak-growth-creating 64 big))
  (printf "memory: 64 unreleased ~a MiB models peaked at ~a MiB of growth; the bound is ~a MiB\n"
          (quotient big MiB)
          (~r (/ peak MiB) #:precision 1)
          (~r (/ bound MiB) #:precision 1))
  (check-true (<= peak bound) (format "peak growth ~a bytes, past the bound ~a" peak bound))
  (drain-finalizers!))
