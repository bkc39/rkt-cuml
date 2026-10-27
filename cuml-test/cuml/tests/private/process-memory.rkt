#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/list make-list)
         (only-in racket/match match-define match-lambda)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until drain-finalizers! settled-count)
         (only-in raft/tests/private/gpu test-unless-skipped)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         (only-in "cuml-error.rkt" raised)
         (only-in "device-memory.rkt" process-device-bytes process-memory-skip))

(provide baseline-sample
         device
         device-growth
         device-slack
         pool-sample
         sample-bytes
         sample-used
         test-process-memory
         test-rounds-and-refusals)

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

(define sample-every 10)

(define (test-rounds-and-refusals what
                                  make-X
                                  one-round
                                  refusals
                                  #:rounds rounds
                                  #:buffers-per-round buffers-per-round)
  (test-process-memory
   (format "~a ~a rounds and their refusals leave device memory flat" rounds what)
   (define X (make-X))
   (define fitted (one-round X))
   (define calls (refusals fitted))
   (define (refuse-all)
     (for ([thunk (in-list calls)])
       (check-pred exn:fail:cuml? (raised thunk))))
   (for ([_ (in-range 2)])
     (one-round X)
     (refuse-all))
   (define failures (settled-count release-failure-count))
   (match-define (list base-used base-bytes) (baseline-sample))
   (define live (jobs-live))
   (define threads (executor-threads))
   (define drops (settled-count buffer-drop-count))
   (define-values (samples refused-drops)
     (for/fold ([samples '()]
                [refused-drops '()]
                #:result (values (reverse samples) (reverse refused-drops)))
               ([_ (in-range (quotient rounds sample-every))])
       (for ([_ (in-range sample-every)])
         (one-round X))
       (define sample (pool-sample))
       (define before (settled-count buffer-drop-count))
       (for ([_ (in-range sample-every)])
         (refuse-all))
       (define after-refusals (pool-sample))
       (check-equal? (sample-used after-refusals)
                     (sample-used sample)
                     "a refused call holds no pool memory once it returns")
       (values (cons sample samples)
               (cons (- (settled-count buffer-drop-count) before) refused-drops))))
   (define wanted (* buffers-per-round rounds))
   (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
   (define freed (- (buffer-drop-count) drops))
   (void/reference-sink X fitted calls)
   (define growth (device-growth base-bytes samples))
   (check-equal? freed wanted)
   (check-equal? refused-drops (make-list (length samples) 0) "refusals hand Racket no buffer to drop")
   (check-equal? (- (release-failure-count) failures) 0)
   (check-true (collect-until (lambda () (= (jobs-live) live))))
   (check-true (<= (executor-threads) (add1 threads)))
   (for ([u (in-list (map sample-used samples))])
     (check-equal? u
                   base-used
                   "pool use returns to the baseline after a collection, so no scratch is kept"))
   (check-true
    (<= growth device-slack)
    (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack))))
