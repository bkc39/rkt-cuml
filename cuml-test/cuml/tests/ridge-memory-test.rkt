#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-predict)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y normal-case on-matrix on-vector)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory))

(define rounds 200)
(define sample-every 20)
(define refusals-per-sample 5)
(define buffers-per-round 2)

(define (data)
  (define c (normal-case 2000 7 0 #:weighted? #t))
  (list (on-matrix (lcase-X c) 'float32)
        (on-vector (lcase-y c) 'float32)
        (on-vector (lcase-w c) 'float32)))

(define (one-round X y w)
  (ridge-predict (ridge-fit X y #:sample-weight w #:alpha 1/2 #:solver 'svd) X))

(define (wide-X columns)
  (define draw (uniform-source 3))
  (on-matrix (for/list ([_ (in-range 2)])
               (for/list ([_ (in-range columns)])
                 (draw)))
             'float32))

(define (mixed-scales)
  (define c (normal-case 60 4 11))
  (list (on-matrix (for/list ([row (in-list (lcase-X c))])
                     (map * row (list (expt 2 -13) 1 (expt 2 13) 1)))
                   'float32)
        (on-vector (lcase-y c) 'float32)))

(define (refusals X y)
  (define wide (wide-X 24000))
  (define wide-y (on-vector '(1 2) 'float32))
  (match-define (list Xm ym) (mixed-scales))
  (list (list "ridge-fit: solver: expected auto, eig or svd"
              (lambda () (ridge-fit X y #:solver 'lsmr)))
        (list "ridge-fit: alpha: expected a finite value at least 0, got -1"
              (lambda () (ridge-fit X y #:alpha -1)))
        (list "ridge-fit: X: 2x24000 elements are past cuSOLVER's eig workspace query"
              (lambda () (ridge-fit wide wide-y)))
        (list #rx"^ridge-fit: X: the eig solve is not a ridge solution"
              (lambda () (ridge-fit Xm ym #:alpha 1 #:solver 'eig)))))

(define (refuse-all calls)
  (for* ([call (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (match-define (list message thunk) call)
    (check-cuml-error 'logic message thunk)))

(test-process-memory (format "~a ridge rounds free every output and leave device memory flat" rounds)
  (match-define (list X y w) (data))
  (define calls (refusals X y))
  (for ([_ (in-range 5)])
    (one-round X y w)
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
        (one-round X y w))
      (define sample (pool-sample))
      (define before (settled-count buffer-drop-count))
      (refuse-all calls)
      (define after-refusals (pool-sample))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused call holds no pool memory once it returns")
      (values (cons sample samples)
              (cons (- (settled-count buffer-drop-count) before) refused-drops))))
  (define wanted (* buffers-per-round rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X y w calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a ridge rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed\n"
   rounds
   (* refusals-per-sample (length calls) (length samples))
   (~r (/ (- (current-inexact-milliseconds) started) 1000.0) #:precision 1)
   base-used
   used
   base-bytes
   (map sample-bytes samples)
   freed
   wanted)
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
