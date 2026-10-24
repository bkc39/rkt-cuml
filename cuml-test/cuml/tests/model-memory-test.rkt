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
                  MiB
                  counts
                  counts-
                  counts-created
                  counts-destroyed
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

(define rounds 200)
(define sample-every 50)
(define models-per-round 4)
(define releases-per-round 3)
(define payload (* 8 MiB))
(define region-length 1024)

(define (model)
  (probe-model #:payload-bytes payload #:region-length region-length))

(define (one-round)
  (call-with-values model void)
  (call-with-values model (lambda (m _region) (native-model-release! m)))
  (call-with-values model (lambda (m _region) (with-native-models ([held m]) (void held))))
  (call-with-values model
                    (lambda (m kept)
                      (native-model-release! m)
                      kept)))

(struct baseline (failures used bytes live threads counts))

(define (warmed-baseline)
  (for ([_ (in-range 5)])
    (one-round))
  (define failures (settled-count release-failure-count))
  (match-define (list used bytes) (baseline-sample))
  (baseline failures used bytes (jobs-live) (executor-threads) (probe-counts)))

(define (sampled-rounds)
  (for/list ([_ (in-range (quotient rounds sample-every))])
    (void/reference-sink (for/list ([_ (in-range sample-every)])
                           (one-round)))
    (pool-sample)))

(define (all-destroyed? before)
  (define now (counts- (probe-counts) before))
  (= (counts-destroyed now) (counts-created now)))

(define (report base samples delta seconds)
  (printf "memory: ~a model rounds in ~a s\n" rounds (~r seconds #:precision 1))
  (printf "memory: pool used ~a before, ~a after\n" (baseline-used base) (map sample-used samples))
  (printf "memory: device ~a before, ~a after\n" (baseline-bytes base) (map sample-bytes samples))
  (match-define (counts created released destroyed stale) delta)
  (printf "memory: ~a created, ~a released, ~a destroyed, ~a stale\n"
          created
          released
          destroyed
          stale))

(test-process-memory "dropped and released models leave device memory flat"
  (define base (warmed-baseline))
  (define started (current-inexact-milliseconds))
  (define samples (sampled-rounds))
  (check-true (collect-until (lambda () (all-destroyed? (baseline-counts base)))))
  (define delta (counts- (probe-counts) (baseline-counts base)))
  (report base samples delta (/ (- (current-inexact-milliseconds) started) 1000.0))
  (check-equal?
   delta
   (counts (* models-per-round rounds) (* releases-per-round rounds) (* models-per-round rounds) 0))
  (check-equal? (- (release-failure-count) (baseline-failures base)) 0)
  (check-true (collect-until (lambda () (= (jobs-live) (baseline-live base)))))
  (check-true (<= (executor-threads) (add1 (baseline-threads base))))
  (for ([used (in-list (map sample-used samples))])
    (check-equal? used (baseline-used base) "pool use returns to the baseline after a collection"))
  (define growth (device-growth (baseline-bytes base) samples))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))

(test-process-memory "a release frees the payload at once, before any collection"
  (define-values (m region) (model))
  (define used (probe-pool-used device))
  (native-model-release! m)
  (check-equal? (- used (probe-pool-used device)) payload)
  (void/reference-sink region))

(define (peak-growth-creating n big)
  (define start (process-device-bytes #:device device))
  (for/fold ([peak 0]) ([_ (in-range n)])
    (call-with-values (lambda () (probe-model #:payload-bytes big #:region-length 1)) void)
    (max peak (- (process-device-bytes #:device device) start))))

(test-process-memory "unreleased models are collected under GC pressure alone"
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
