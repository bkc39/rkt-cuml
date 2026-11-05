#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit linear-regression-predict)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
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

(define rounds 50)
(define sample-every 10)
(define refusals-per-sample 5)
(define buffers-per-round 2)

(define (data)
  (define c (normal-case 2000 7 0 #:weighted? #t))
  (list (on-matrix (lcase-X c) 'float32)
        (on-vector (lcase-y c) 'float32)
        (on-vector (lcase-w c) 'float32)
        (on-matrix '((1 2) (3 4)) 'float32)))

(define (one-round X y w)
  (linear-regression-predict (linear-regression-fit X y #:sample-weight w #:algorithm 'qr) X))

(define (refusals X y X2)
  (define nan-X (on-matrix (cons '(+nan.0 0 0 0 0 0 0) (make-list 1999 '(1 1 1 1 1 1 1))) 'float32))
  (define no-weight (on-vector (make-list 2000 0) 'float32))
  (define fitted (linear-regression-fit X y))
  (list (list "linear-regression-fit: X: row 0, column 0: expected a finite value, got nan"
              (lambda () (linear-regression-fit nan-X y)))
        (list "linear-regression-fit: sample-weight: expected a positive sum, got 0"
              (lambda () (linear-regression-fit X y #:sample-weight no-weight)))
        (list "linear-regression-predict: X: expected 7 columns, got 2"
              (lambda () (linear-regression-predict fitted X2)))))

(define (refuse-all calls)
  (for* ([call (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (match-define (list message thunk) call)
    (check-cuml-error 'logic message thunk)))

(test-process-memory (format
                      "~a rounds of linear regression free every output and leave device memory flat"
                      rounds)
  (match-define (list X y w X2) (data))
  (define calls (refusals X y X2))
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
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused call holds no pool memory once it returns")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* buffers-per-round rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X y w X2 calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a linear-regression rounds and ~a refused calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
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
