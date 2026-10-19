#lang racket/base

(require (only-in cuml agglomerative-clustering exn:fail:cuml? fit-predict)
         (only-in cuml/cluster single-linkage-fit)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/dbscan-data.rkt" integer-grid mat)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/linkage-data.rkt" group-rows)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory))

(define rounds 300)
(define sample-every 100)
(define refusals-per-sample 20)
(define fits-per-round 6)
(define buffers-per-fit 2)
(define n-samples 1500)

(define (data)
  (list (mat (integer-grid n-samples 2 40 6) 'float32)
        (mat (integer-grid 50 2 10 7) 'float32)
        (mat '((1 2)) 'float32)
        (mat (group-rows 1024 32) 'float32)))

(define (one-round X small one e)
  (single-linkage-fit X #:connectivity 'pairwise)
  (single-linkage-fit X #:n-clusters 5)
  (single-linkage-fit small #:n-clusters 1)
  (single-linkage-fit small #:n-clusters 50 #:connectivity 'pairwise)
  (single-linkage-fit one #:n-clusters 1)
  (fit-predict e X))

(define (refusals X)
  (define wide (blank-matrix n-samples 2 'float64))
  (define no-columns (blank-matrix n-samples 0 'float32))
  (define pairwise-rows (blank-matrix 46341 1 'float32))
  (define nan-row (mat '((1 1) (2 2) (+nan.0 0)) 'float32))
  (list (lambda () (single-linkage-fit wide))
        (lambda () (single-linkage-fit X #:metric 'cosine))
        (lambda () (single-linkage-fit no-columns))
        (lambda () (single-linkage-fit X #:n-clusters 0))
        (lambda () (single-linkage-fit pairwise-rows #:connectivity 'pairwise))
        (lambda () (single-linkage-fit nan-row))))

(define (refuse-all calls groups)
  (for* ([thunk (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (check-pred exn:fail:cuml? (raised thunk)))
  (check-pred exn:fail:cuml? (raised (lambda () (single-linkage-fit groups)))))

(test-process-memory (format
                      "~a rounds of single-linkage fits free both buffers of each fit and leave device memory flat"
                      rounds)
  (match-define (list X small one groups) (data))
  (define e (agglomerative-clustering #:n-clusters 4 #:connectivity 'pairwise))
  (define calls (refusals X))
  (for ([_ (in-range 5)])
    (one-round X small one e)
    (refuse-all calls groups))
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
        (one-round X small one e))
      (define sample (pool-sample))
      (define before (settled-count buffer-drop-count))
      (refuse-all calls groups)
      (define after-refusals (pool-sample))
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused or failed call holds no pool memory once it returns")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* fits-per-round buffers-per-fit rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X small one groups e calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a single-linkage rounds and ~a refused or failed calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
   rounds
   (* (add1 (* refusals-per-sample (length calls))) (length samples))
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
    (check-equal? u base-used "pool use returns to the baseline after a collection"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))
