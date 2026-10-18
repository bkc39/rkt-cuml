#lang racket/base

(require (only-in cuml dbscan exn:fail:cuml? fit-predict)
         (only-in cuml/cluster dbscan-fit)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/dbscan-data.rkt" distance-matrix integer-grid mat vec)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
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
  (define rows (map (lambda (r) (map add1 r)) (integer-grid n-samples 2 40 6)))
  (list (mat rows 'float32)
        (mat (distance-matrix (integer-grid 400 2 20 7)) 'float64)
        (vec (for/list ([i (in-range n-samples)])
               (add1 (modulo i 3)))
             'float32)))

(define (one-round X D w e)
  (dbscan-fit X #:eps 1.5 #:min-samples 4)
  (dbscan-fit X #:eps 0.01 #:min-samples 4 #:metric 'cosine)
  (dbscan-fit D #:eps 1.5 #:min-samples 4 #:metric 'precomputed #:max-mbytes-per-batch 1)
  (dbscan-fit X #:eps 1.5 #:min-samples 6 #:sample-weight w)
  (dbscan-fit X #:eps 1.5 #:min-samples 4 #:out-dtype 'int64)
  (fit-predict e X))

(define (refusals X)
  (define zero-row (mat '((1 1) (2 2) (0 0)) 'float32))
  (define wide-weights (blank-vector n-samples 'float64))
  (define col-major (blank-matrix n-samples 2 'float32 #:layout 'col-major))
  (define crashing (blank-matrix 150000 2 'float32))
  (list (lambda () (dbscan-fit X #:eps 0))
        (lambda () (dbscan-fit X #:sample-weight wide-weights))
        (lambda () (dbscan-fit col-major))
        (lambda () (dbscan-fit X #:metric 'precomputed))
        (lambda () (dbscan-fit X #:min-samples 0))
        (lambda () (dbscan-fit crashing #:max-mbytes-per-batch 1))
        (lambda () (dbscan-fit zero-row #:metric 'cosine))))

(define (refuse-all calls)
  (for* ([thunk (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (check-pred exn:fail:cuml? (raised thunk))))

(test-process-memory (format
                      "~a rounds of DBSCAN fits free both buffers of each fit and leave device memory flat"
                      rounds)
  (match-define (list X D w) (data))
  (define e (dbscan #:eps 1.5 #:min-samples 4))
  (define calls (refusals X))
  (for ([_ (in-range 5)])
    (one-round X D w e)
    (refuse-all calls))
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
        (one-round X D w e))
      (define sample (pool-sample))
      (define before (settled-count buffer-drop-count))
      (refuse-all calls)
      (define after-refusals (pool-sample))
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused call holds no pool memory once it returns")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* fits-per-round buffers-per-fit rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X D w e calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a DBSCAN rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
   rounds
   (* refusals-per-sample (length calls) (length samples))
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
