#lang racket/base

(require (only-in cuml/core/private/model deposit->native-model live-anchor)
         (only-in cuml/core/private/native cuml-family-call cuml-family-library load-family)
         (only-in ffi/unsafe _fun _int _int32 _int64 _ptr get-ffi-obj)
         (only-in racket/match match-define-values)
         (only-in racket/path path-only)
         (only-in rackunit fail)
         (only-in raft current-device-resources)
         (only-in raft/tests/private/gpu gpu-skip test-unless-skipped)
         (only-in raft/unsafe deposit define-job)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define)

(provide (struct-out counts)
         await-waiter
         check-model-probe
         probe-counts
         probe-create-job
         probe-model
         probe-release
         probe-touch
         probe-touch-job
         test-model-probe
         with-gate)

(define probe-path (getenv "CUML_SHIM_PROBE"))

(define probe (and probe-path (load-family "probe" (path-only probe-path))))

(define probe-missing
  "no model probe library (CUML_SHIM_PROBE is not set; run inside nix develop)")

(define library (and probe (cuml-family-library probe)))

(define (entry name type)
  (if library
      (get-ffi-obj name library type)
      (lambda _ (error 'model-probe probe-missing))))

(define-syntax-parse-rule (define-probe-job id:id c-id:str params:expr)
  (define id
    (if probe
        (let ()
          (define-job id library c-id params)
          id)
        (lambda _ (error 'id probe-missing)))))

(define-probe-job probe-create-job "cuml_probe_model_create_job" (handle deposit _int64 _int32 _int32))
(define-probe-job probe-touch-job "cuml_probe_model_touch_job" (handle view _int32 (out _int64)))
(define-probe-job probe-release-job "cuml_probe_model_release_job" (handle view))

(define counts-entry
  (entry "cuml_probe_model_counts"
         (_fun (created : (_ptr o _int64))
               (released : (_ptr o _int64))
               (destroyed : (_ptr o _int64))
               (stale : (_ptr o _int64))
               ->
               (status : _int)
               ->
               (values status created released destroyed stale))))

(define gate-create-entry
  (entry "cuml_probe_gate_create" (_fun (gate : (_ptr o _int32)) -> (status : _int) -> (values status gate))))

(define gate-open-entry (entry "cuml_probe_gate_open" (_fun _int32 -> _int)))

(define gate-waiters-entry
  (entry "cuml_probe_gate_waiters"
         (_fun _int32 (waiters : (_ptr o _int32)) -> (status : _int) -> (values status waiters))))

(define (call who thunk)
  ((cuml-family-call probe) who thunk))

(struct counts (created released destroyed stale) #:transparent)

(define (probe-counts)
  (define-values (created released destroyed stale) (call 'probe-counts counts-entry))
  (counts created released destroyed stale))

(define kind-codes (hasheq 'probe-a 1 'probe-b 2))

(define (probe-release who resources anchor)
  (probe-release-job who resources anchor))

(define (probe-model #:kind [kind 'probe-a]
                     #:payload-bytes [payload-bytes 0]
                     #:region-length [region-length 4]
                     #:release [release probe-release]
                     #:resources [resources (current-device-resources)])
  (define out (deposit #:resources resources))
  (probe-create-job 'probe-model
                    resources
                    out
                    payload-bytes
                    region-length
                    (hash-ref kind-codes kind kind))
  (match-define-values (m (list region))
                       (deposit->native-model 'probe-model
                                              kind
                                              out
                                              release
                                              #:opaque-bytes payload-bytes))
  (values m region))

(define (probe-touch m #:gate [gate -1] #:resources [resources (current-device-resources)])
  (probe-touch-job 'probe-touch resources (live-anchor 'probe-touch m 'probe-a) gate))

(define (await-waiter gate [seconds 5])
  (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
  (let poll ()
    (cond
      [(positive? (call 'await-waiter (lambda () (gate-waiters-entry gate)))) #t]
      [(> (current-inexact-milliseconds) deadline) #f]
      [else
       (sleep 0.01)
       (poll)])))

(define (call-with-gate proc)
  (define gate (call 'with-gate gate-create-entry))
  (dynamic-wind void
                (lambda () (proc gate (lambda () (call 'with-gate (lambda () (gate-open-entry gate))))))
                (lambda () (call 'with-gate (lambda () (gate-open-entry gate))))))

(define-syntax-parse-rule (with-gate (gate:id open!:id) body:expr ...+)
  (call-with-gate (lambda (gate open!)
                    body ...)))

(define (check-model-probe)
  (unless probe
    (fail probe-missing)))

(define-syntax-parse-rule (test-model-probe name:expr body:expr ...+)
  (test-unless-skipped gpu-skip name
    (check-model-probe)
    body ...))
