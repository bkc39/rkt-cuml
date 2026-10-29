#lang racket/base

(require (only-in cuml/metrics
                  adjusted-rand-score
                  contingency-matrix
                  silhouette-samples
                  silhouette-score
                  trustworthiness)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list range)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft list*->device-matrix)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/metrics-data.rkt" label-pair labels-of overlap overlap-labels projection)
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
(define buffers-per-round 4)

(define (data)
  (define-values (truth found) (label-pair 2000 6 9 3))
  (define-values (X E) (projection 300 10 2))
  (list (labels-of truth 'int32)
        (labels-of found 'int64)
        (overlap 'float32 #:each 100)
        (labels-of (overlap-labels #:each 100) 'int32)
        X
        E))

(define (one-round truth found S ys X E)
  (contingency-matrix truth found)
  (adjusted-rand-score truth found)
  (silhouette-score S ys)
  (silhouette-samples S ys #:metric 'cosine)
  (trustworthiness X E))

(define (refusals truth S ys X)
  (define short (labels-of '(0 1 2) 'int32))
  (define many (labels-of (range 46341) 'int32))
  (define flat (labels-of (make-list 46341 0) 'int32))
  (define one-label (labels-of (make-list 400 3) 'int32))
  (define nan-rows (list*->device-matrix (cons '(+nan.0 0) (make-list 399 '(1 1))) #:dtype 'float32))
  (define big-rows (list*->device-matrix (cons '(1e20 0) (make-list 299 '(1 1))) #:dtype 'float32))
  (define float-most "4.253529586511731e+37")
  (list
   (list "contingency-matrix: labels-pred: expected 2000 elements, got 3"
         (lambda () (contingency-matrix truth short)))
   (list "adjusted-rand-score: labels-true: expected at most 46340 distinct labels, got 46341"
         (lambda () (adjusted-rand-score many flat)))
   (list "silhouette-score: labels: expected at least 2 distinct labels, got 1"
         (lambda () (silhouette-score S one-label)))
   (list (format "silhouette-samples: X: row 0: expected a squared norm of at most ~a, got nan"
                 float-most)
         (lambda () (silhouette-samples nan-rows ys)))
   (list "trustworthiness: n-neighbors: expected less than half the 300 rows, got 150"
         (lambda () (trustworthiness X X #:n-neighbors 150)))
   (list (format "trustworthiness: X-embedded: row 0: expected a squared norm of at most ~a, got ~a"
                 float-most
                 "1.0000000400817551e+40")
         (lambda () (trustworthiness X big-rows)))))

(define (refuse-all calls)
  (for* ([call (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (match-define (list message thunk) call)
    (check-cuml-error 'logic message thunk)))

(test-process-memory (format "~a rounds of the metrics free every output and leave device memory flat"
                             rounds)
  (match-define (list truth found S ys X E) (data))
  (define calls (refusals truth S ys X))
  (for ([_ (in-range 5)])
    (one-round truth found S ys X E)
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
        (one-round truth found S ys X E))
      (define sample (pool-sample))
      (define before (settled-count buffer-drop-count))
      (refuse-all calls)
      (define after-refusals (pool-sample))
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused call holds no pool memory once it returns")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* buffers-per-round rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink truth found S ys X E calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a metrics rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
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

(test-process-memory "a contingency table holds its counts and labels, not the n-sized relabel scratch"
  (define n (expt 2 22))
  (define a
    (labels-of (for/list ([i (in-range n)])
                 (modulo i 2))
               'int64))
  (define b
    (labels-of (for/list ([i (in-range n)])
                 (modulo i 3))
               'int64))
  (define before (sample-used (pool-sample)))
  (define table (contingency-matrix a b))
  (define held (- (sample-used (pool-sample)) before))
  (define output (* 8 (+ (* 2 3) 2 3)))
  (printf "memory: a 2x3 table over ~a labels holds ~a pool bytes; its output is ~a\n" n held output)
  (check-true (<= held (+ output (* 1024 1024)))
              (format "the table holds ~a bytes, past its ~a-byte output and 1 MiB" held output))
  (void/reference-sink a b table))
