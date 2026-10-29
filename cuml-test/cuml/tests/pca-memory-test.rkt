#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in cuml/decomposition pca-fit pca-inverse-transform pca-transform)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory)
         (only-in "private/scaler-data.rkt" normal-rows rows->matrix))

(define rounds 50)
(define sample-every 10)
(define settings '((full #f) (full #t) (jacobi #f) (jacobi #t)))
(define buffers-per-round (* 7 (length settings)))

(define (one-round X)
  (for/last ([setting (in-list settings)])
    (match-define (list solver whiten) setting)
    (define fitted (pca-fit X #:n-components 8 #:solver solver #:whiten whiten))
    (pca-inverse-transform fitted (pca-transform fitted X))
    fitted))

(define (refusals fitted)
  (define nan-row (rows->matrix (cons (list +nan.0 1.0) (make-list 99 '(1.0 2.0))) 'float32))
  (define no-rows (blank-matrix 0 32 'float32))
  (define narrow (blank-matrix 4 3 'float32))
  (define integers (blank-matrix 4 32 'int32))
  (define inf-row (rows->matrix (list (cons +inf.0 (make-list 31 0.0))) 'float32))
  (define one-row (blank-matrix 1 32 'float32))
  (define eight-rows (blank-matrix 8 32 'float32))
  (list (lambda () (pca-fit nan-row))
        (lambda () (pca-fit one-row))
        (lambda () (pca-fit eight-rows #:n-components 40))
        (lambda () (pca-transform fitted narrow))
        (lambda () (pca-inverse-transform fitted integers))
        (lambda () (pca-transform fitted inf-row))
        (lambda () (pca-transform fitted no-rows))))

(test-process-memory (format
                      "~a fit, transform and inverse rounds and their refusals leave device memory flat"
                      rounds)
  (define X (rows->matrix (normal-rows 7 4096 32 'float32) 'float32))
  (define fitted (one-round X))
  (define calls (refusals fitted))
  (define (refuse-all)
    (for ([thunk (in-list calls)])
      (check-pred exn:fail:cuml? (raised thunk))))
  (for ([_ (in-range 3)])
    (one-round X)
    (refuse-all))
  (define failures (settled-count release-failure-count))
  (match-define (list base-used base-bytes) (baseline-sample))
  (define live (jobs-live))
  (define threads (executor-threads))
  (define drops (settled-count buffer-drop-count))
  (define started (current-inexact-milliseconds))
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
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a pca rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
   rounds
   (* sample-every (length calls) (length samples))
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   base-used
   used
   base-bytes
   (map sample-bytes samples)
   freed
   wanted
   refused-drops)
  (check-equal? freed wanted)
  (check-equal? refused-drops (make-list (length samples) 0) "refusals hand Racket no buffer to drop")
  (check-equal? (- (release-failure-count) failures) 0)
  (check-true (collect-until (lambda () (= (jobs-live) live))))
  (check-true (<= (executor-threads) (add1 threads)))
  (for ([u (in-list used)])
    (check-equal? u
                  base-used
                  "pool use returns to the baseline after a collection, so no scratch is kept"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))
