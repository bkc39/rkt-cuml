#lang racket/base

(require (only-in cuml/cluster kmeans-fit kmeans-predict kmeans-result kmeans-score kmeans-transform)
         (only-in cuml/cluster/private/blobs blobs)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft device-resources device-vector->list shape with-device-resources)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" raised)
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
(define refusals-per-sample 30)
(define buffers-per-round 4)
(define n-samples 65536)
(define n-features 4)
(define n-clusters 8)

(define (data)
  (define-values (X _labels) (blobs n-samples n-features #:centers n-clusters #:seed 9))
  X)

(define (one-round X)
  (match-define (kmeans-result centroids _ _ _)
    (kmeans-fit X #:n-clusters n-clusters #:max-iter 20 #:random-state 1))
  (kmeans-predict centroids X)
  (kmeans-transform centroids X)
  (kmeans-score centroids X))

(define (refuse X)
  (raised (lambda () (kmeans-fit X #:n-clusters (add1 n-samples)))))

(test-process-memory "300 rounds of fit, predict, transform and score free every buffer and leave device memory flat"
  (define X (data))
  (for ([_ (in-range 5)])
    (one-round X)
    (refuse X))
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
      (for ([_ (in-range refusals-per-sample)])
        (refuse X))
      (define after-refusals (pool-sample))
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused fit holds no pool memory")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* buffers-per-round rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a k-means rounds of ~ax~a (k ~a) in ~a s, ~a refused fits; pool used ~a bytes before, ~a after each ~a; this process's device memory ~a bytes before, ~a after each ~a; ~a buffers freed of ~a allocated, ~a by refusals; ~a release failures\n"
   rounds
   n-samples
   n-features
   n-clusters
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   (* refusals-per-sample (length samples))
   base-used
   used
   sample-every
   base-bytes
   (map sample-bytes samples)
   sample-every
   freed
   wanted
   refused-drops
   (- (release-failure-count) failures))
  (check-equal? freed wanted)
  (check-equal? refused-drops
                (make-list (length samples) 0)
                "refusals free nothing: they allocate nothing")
  (check-equal? (- (release-failure-count) failures) 0)
  (check-true (collect-until (lambda () (= (jobs-live) live))))
  (check-true (<= (executor-threads) (add1 threads)))
  (for ([u (in-list used)])
    (check-equal? u base-used "pool use returns to the baseline after a collection"))
  (check-true
   (<= growth device-slack)
   (format "this process's device memory grew ~a bytes, past the ~a slack" growth device-slack)))

(test-gpu "a fit on its own device resources gives usable outputs"
  (with-device-resources ([r (device-resources)])
    (define-values (points _labels) (blobs 600 3 #:centers 3 #:seed 4 #:resources r))
    (match-define (kmeans-result centroids labels inertia _)
      (kmeans-fit points #:n-clusters 3 #:random-state 1 #:resources r))
    (check-equal? (shape centroids) '(3 3))
    (check-equal? (device-vector->list (kmeans-predict centroids points #:resources r))
                  (device-vector->list labels))
    (check-equal? (kmeans-score centroids points #:resources r) (- inertia))
    (check-equal? (shape (kmeans-transform centroids points #:resources r)) '(600 3))))
