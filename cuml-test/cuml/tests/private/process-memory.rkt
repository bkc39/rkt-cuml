#lang racket/base

(require (only-in racket/match match-define match-lambda)
         (only-in rackunit check-true)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-unless-skipped)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         (only-in "device-memory.rkt" process-device-bytes process-memory-skip))

(provide baseline-sample
         device
         device-growth
         device-slack
         pool-sample
         sample-bytes
         sample-used
         test-process-memory)

(define device 0)
(define device-slack (* 128 1024 1024))

(define (pool-sample)
  (drain-finalizers!)
  (list (probe-pool-used device) (process-device-bytes #:device device)))

(define (baseline-sample)
  (match-define (and sample (list _ bytes)) (pool-sample))
  (check-true (positive? bytes) "NVML lists this process once it holds a CUDA context")
  sample)

(define sample-used
  (match-lambda
    [(list used _) used]))

(define sample-bytes
  (match-lambda
    [(list _ bytes) bytes]))

(define (device-growth base-bytes samples)
  (- (apply max (map sample-bytes samples)) base-bytes))

(define-syntax-parse-rule (test-process-memory name:expr body:expr ...+)
  (test-pools name
    (test-unless-skipped (process-memory-skip #:device device) name body ...)))
