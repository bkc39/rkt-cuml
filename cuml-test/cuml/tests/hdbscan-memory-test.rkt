#lang racket/base

(require (only-in cuml exn:fail:cuml? fit-predict hdbscan native-model-release!)
         (only-in cuml/cluster
                  hdbscan-all-points-membership-vectors
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-membership-vector
                  hdbscan-result-model)
         (only-in cuml/cluster/private/deposit deposit-for/who)
         (only-in cuml/cluster/private/native
                  cuml-cluster-hdbscan-fit-job
                  cuml-cluster-model-release-job)
         (only-in cuml/core/private/model deposit->native-model/who)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft current-device-resources device-vector->list)
         (only-in raft/unsafe deposit->list)
         (only-in raft/private/foreign/memory release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until drain-finalizers! settled-count)
         (only-in raft/tests/private/probe probe-pool-used)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/hdbscan-data.rkt" dataset)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory))

(define rounds 300)
(define sample-every 100)
(define refusals-per-sample 20)

(define (one-round i X points e)
  (define result (hdbscan-fit X))
  (hdbscan-approximate-predict result points)
  (hdbscan-membership-vector result points)
  (hdbscan-all-points-membership-vectors result)
  (fit-predict e X)
  (when (even? i)
    (native-model-release! (hdbscan-result-model result))))

(define (refusals X points)
  (define released (hdbscan-fit X))
  (native-model-release! (hdbscan-result-model released))
  (define live (hdbscan-fit X))
  (define short
    (hdbscan-fit (mat (take (dataset 'b50) 9) 'float32) #:min-cluster-size 2 #:min-samples 5))
  (define nan-row (mat '((1 1) (2 2) (+nan.0 0) (4 4)) 'float32))
  (list (lambda () (hdbscan-fit (blank-matrix 300 2 'float64)))
        (lambda () (hdbscan-fit (blank-matrix 1 2 'float32)))
        (lambda () (hdbscan-fit X #:min-cluster-size 1))
        (lambda () (hdbscan-fit (mat '((1 1) (2 2) (+nan.0 0) (4 4) (5 5) (6 6)) 'float32)))
        (lambda () (hdbscan-approximate-predict live (blank-matrix 0 2 'float32)))
        (lambda () (hdbscan-membership-vector live (blank-matrix 4 3 'float32)))
        (lambda () (hdbscan-approximate-predict short points))
        (lambda () (hdbscan-membership-vector live nan-row))
        (lambda () (hdbscan-approximate-predict released points))))

(define (refuse-all calls)
  (for* ([thunk (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (check-pred exn:fail:cuml? (raised thunk))))

(test-process-memory (format
                      "~a rounds of HDBSCAN fits, predictions and memberships leave device memory flat"
                      rounds)
  (define X (mat (dataset 'b50) 'float32))
  (define points (mat (dataset 'b10) 'float32))
  (define e (hdbscan))
  (define calls (refusals X points))
  (for ([i (in-range 4)])
    (one-round i X points e)
    (refuse-all calls))
  (define failures (settled-count release-failure-count))
  (match-define (list base-used base-bytes) (baseline-sample))
  (define live (jobs-live))
  (define threads (executor-threads))
  (define started (current-inexact-milliseconds))
  (define samples
    (for/list ([_ (in-range (quotient rounds sample-every))])
      (for ([i (in-range sample-every)])
        (one-round i X points e))
      (define sample (pool-sample))
      (refuse-all calls)
      (check-equal? (sample-used (pool-sample))
                    (sample-used sample)
                    "a refused call holds no pool memory once it returns")
      sample))
  (void/reference-sink X points e calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a HDBSCAN rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after\n"
   rounds
   (* refusals-per-sample (length calls) (length samples))
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   base-used
   used
   base-bytes
   (map sample-bytes samples))
  (check-equal? (- (release-failure-count) failures) 0)
  (check-true (collect-until (lambda () (= (jobs-live) live))))
  (check-true (<= (executor-threads) (add1 threads)))
  (for ([u (in-list used)])
    (check-equal? u base-used "pool use returns to the baseline after a collection"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))

(define (opaque-bounds n regions)
  (match-define (list _ parents _ _ _ _ selected) regions)
  (define k (length (device-vector->list selected)))
  (define edges (length (device-vector->list parents)))
  (define clusters (- (add1 edges) n))
  (define least (+ (* 4 n) (* 8 (+ (add1 k) k (add1 edges))) (* 4 clusters)))
  (values least (+ least (* 8 n))))

(test-process-memory "a release frees exactly the opaque bytes the fit reports"
  (define rows (dataset 'overlap))
  (define resources (current-device-resources))
  (define model-out (deposit-for/who 'opaque resources))
  (define out (deposit-for/who 'opaque resources))
  (define X (mat rows 'float32))
  (define opaque (cuml-cluster-hdbscan-fit-job 'opaque resources X model-out out 5 5 0.0 0 1.0 0 0))
  (define-values (model regions)
    (deposit->native-model/who 'opaque
                               'hdbscan
                               model-out
                               cuml-cluster-model-release-job
                               #:opaque-bytes opaque))
  (define plain (deposit->list out))
  (define-values (least most) (opaque-bounds (length rows) regions))
  (drain-finalizers!)
  (define used (probe-pool-used device))
  (native-model-release! model)
  (define freed (- used (probe-pool-used device)))
  (printf "memory: a release freed ~a bytes; the fit reported ~a, in ~a to ~a\n" freed opaque least most)
  (check-equal? freed opaque)
  (check-true (<= least opaque most))
  (void/reference-sink X regions plain))
