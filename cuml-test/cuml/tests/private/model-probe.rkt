#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/core exn:fail:cuml)
         (only-in cuml/core/private/hyperparameters define-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in cuml/core/private/model deposit->native-model/who live-anchor/who)
         (only-in cuml/core/private/native cuml-family-call cuml-family-library load-family)
         (only-in cuml/core/private/release call-with-release/who)
         (only-in ffi/unsafe _fun _int _int32 _int64 _ptr get-ffi-obj)
         (only-in racket/match match-define match-define-values)
         (only-in racket/path path-only)
         (only-in raft current-device-resources)
         (only-in raft/tests/private/gpu gpu-skip test-unless-skipped)
         (only-in raft/unsafe deposit define-job)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define)

(provide (struct-out counts)
         MiB
         await-waiter
         counts-
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

(define library (and probe (cuml-family-library probe)))

(define (missing . _arguments)
  (logic-error
   "model-probe: no model probe library (CUML_SHIM_PROBE is not set; run inside nix develop)"))

(define-syntax-parse-rule (define-probe-entry id:id
                                              c-id:str
                                              (~or* (~seq #:job params:expr) type:expr))
  (define id
    (cond
      [(not library) missing]
      [~?
       [else (define-job id library c-id params #:exn exn:fail:cuml) id]
       [else (get-ffi-obj c-id library type)]])))

(define-probe-entry probe-create-job
                    "cuml_probe_model_create_job"
                    #:job (handle deposit _int64 _int32 _int32))
(define-probe-entry probe-touch-job
                    "cuml_probe_model_touch_job"
                    #:job (handle view _int32 (out _int64)))
(define-probe-entry probe-release "cuml_probe_model_release_job" #:job (handle view))
(define-probe-entry counts-entry
                    "cuml_probe_model_counts"
                    (_fun (created : (_ptr o _int64))
                          (released : (_ptr o _int64))
                          (destroyed : (_ptr o _int64))
                          (stale : (_ptr o _int64))
                          -> (status : _int)
                          -> (values status created released destroyed stale)))
(define-probe-entry gate-create-entry
                    "cuml_probe_gate_create"
                    (_fun (gate : (_ptr o _int32)) -> (status : _int) -> (values status gate)))
(define-probe-entry gate-open-entry "cuml_probe_gate_open" (_fun _int32 -> _int))
(define-probe-entry
 gate-waiters-entry
 "cuml_probe_gate_waiters"
 (_fun _int32 (waiters : (_ptr o _int32)) -> (status : _int) -> (values status waiters)))

(define call
  (if probe
      (cuml-family-call probe)
      missing))

(struct counts (created released destroyed stale) #:transparent)

(define (probe-counts)
  (define-values (created released destroyed stale) (call 'probe-counts counts-entry))
  (counts created released destroyed stale))

(define (counts- after before)
  (match-define (counts c r d s) after)
  (match-define (counts c0 r0 d0 s0) before)
  (counts (- c c0) (- r r0) (- d d0) (- s s0)))

(define MiB (* 1024 1024))

(define-codes kind-code [probe-a 1] [probe-b 2])

(define (probe-model #:kind [kind 'probe-a]
                     #:payload-bytes [payload-bytes 0]
                     #:region-length [region-length 4]
                     #:release [release probe-release]
                     #:resources [resources (current-device-resources)])
  (define out (deposit #:resources resources))
  (probe-create-job 'probe-model resources out payload-bytes region-length (kind-code kind))
  (match-define-values (m (list region))
    (deposit->native-model/who 'probe-model kind out release #:opaque-bytes payload-bytes))
  (values m region))

(define (probe-touch m #:gate [gate -1] #:resources [resources (current-device-resources)])
  (probe-touch-job 'probe-touch resources (live-anchor/who 'probe-touch m 'probe-a) gate))

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
  (define (open!)
    (call 'with-gate (lambda () (gate-open-entry gate))))
  (call-with-release/who 'with-gate gate (lambda (_shut) (open!)) (lambda (held) (proc held open!))))

(define-syntax-parse-rule (with-gate (gate:id open!:id) body:expr ...+)
  (call-with-gate (lambda (gate open!)
                    body ...)))

(define-syntax-parse-rule (test-model-probe name:expr body:expr ...+)
  (test-unless-skipped gpu-skip name body ...))
