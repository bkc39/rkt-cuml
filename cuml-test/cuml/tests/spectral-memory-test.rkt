#lang racket/base

(require (only-in cuml exn:fail:cuml? fit-predict spectral-clustering)
         (only-in cuml/cluster spectral-clustering-fit spectral-clustering-fit-graph)
         (only-in ffi/unsafe void/reference-sink)
         (only-in racket/format ~r)
         (only-in racket/list list-set make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred check-true)
         (only-in raft/private/foreign/memory buffer-drop-count release-failure-count)
         (only-in raft/private/job executor-threads jobs-live)
         (only-in raft/tests/private/collect collect-until settled-count)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/process-memory.rkt"
                  baseline-sample
                  device-growth
                  device-slack
                  pool-sample
                  sample-bytes
                  sample-used
                  test-process-memory)
         (only-in "private/spectral-data.rkt"
                  blobs
                  cliques
                  graph
                  graph->vectors
                  graph-cols
                  graph-n
                  graph-rows
                  graph-vals
                  sparse-blocks))

(define rounds 120)
(define sample-every 40)
(define refusals-per-sample 20)
(define fits-per-round 5)
(define buffers-per-fit 1)

(define (data)
  (define-values (rows _truth) (blobs 100 3 2 50.0 0))
  (define-values (small _small-truth) (blobs 5 2 2 50.0 1))
  (define-values (g _gtruth) (sparse-blocks '(30 40 50) 0.3 0))
  (define-values (uneven _uneven-truth) (cliques '(5 17 30)))
  (list (mat rows 'float32) (mat small 'float32) g uneven))

(define ((fit-graph vectors n k))
  (match-define (list rows cols vals) vectors)
  (spectral-clustering-fit-graph rows cols vals n #:n-clusters k #:random-state 0))

(define (one-round X small vectors n e)
  (match-define (list rows cols vals) vectors)
  (spectral-clustering-fit X #:n-clusters 3)
  (spectral-clustering-fit-graph rows cols vals n #:n-clusters 3)
  (spectral-clustering-fit X #:n-clusters 1)
  (spectral-clustering-fit small #:n-clusters 2 #:n-neighbors 50)
  (fit-predict e X))

(define (edited g #:rows [rows values] #:cols [cols values] #:vals [vals values])
  (graph->vectors
   (graph (rows (graph-rows g)) (cols (graph-cols g)) (vals (graph-vals g)) (graph-n g))))

(define (refusals X g)
  (define wide (blank-matrix 300 2 'float64))
  (define three (blank-matrix 3 2 'float32))
  (define nan-row (mat '((1 1) (2 2) (+nan.0 0) (4 4) (5 5)) 'float32))
  (match-define (list rows _ vals) (graph->vectors g))
  (define short-cols (blank-vector (sub1 (length (graph-rows g))) 'int32))
  (define out-of-range (edited g #:rows (lambda (xs) (list-set xs 7 120))))
  (define self-loop
    (edited g #:rows (lambda (xs) (list-set xs 7 5)) #:cols (lambda (xs) (list-set xs 7 5))))
  (define negative (edited g #:vals (lambda (xs) (list-set xs 3 -1.0))))
  (define zeros (edited g #:vals (lambda (xs) (map (lambda (_) 0.0) xs))))
  (define (on vs)
    (fit-graph vs (graph-n g) 3))
  (list (lambda () (spectral-clustering-fit wide))
        (lambda () (spectral-clustering-fit three #:n-clusters 2))
        (lambda () (spectral-clustering-fit X #:n-components 0))
        (lambda () (spectral-clustering-fit nan-row #:n-clusters 2))
        (lambda () (spectral-clustering-fit-graph rows short-cols vals (graph-n g)))
        (on out-of-range)
        (on self-loop)
        (on negative)
        (on zeros)))

(define (refuse-all calls unconverged)
  (for* ([thunk (in-list calls)]
         [_ (in-range refusals-per-sample)])
    (check-pred exn:fail:cuml? (raised thunk)))
  (check-pred exn:fail:cuml? (raised unconverged)))

(test-process-memory (format
                      "~a rounds of spectral fits free the labels of each fit and leave device memory flat"
                      rounds)
  (match-define (list X small g uneven) (data))
  (define vectors (graph->vectors g))
  (define n (graph-n g))
  (define e (spectral-clustering #:n-clusters 3))
  (define calls (refusals X g))
  (define unconverged (fit-graph (graph->vectors uneven) (graph-n uneven) 3))
  (for ([_ (in-range 5)])
    (one-round X small vectors n e)
    (refuse-all calls unconverged))
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
        (one-round X small vectors n e))
      (define sample (pool-sample))
      (define before (settled-count buffer-drop-count))
      (refuse-all calls unconverged)
      (define after-refusals (pool-sample))
      (define refused (- (settled-count buffer-drop-count) before))
      (check-equal? (sample-used after-refusals)
                    (sample-used sample)
                    "a refused or failed call holds no pool memory once it returns")
      (values (cons sample samples) (cons refused refused-drops))))
  (define wanted (* fits-per-round buffers-per-fit rounds))
  (check-true (collect-until (lambda () (>= (buffer-drop-count) (+ drops wanted)))))
  (define freed (- (buffer-drop-count) drops))
  (void/reference-sink X small vectors unconverged e calls)
  (define used (map sample-used samples))
  (define growth (device-growth base-bytes samples))
  (printf
   "memory: ~a spectral rounds and ~a refused or failed calls in ~a s; pool used ~a before, ~a after; device ~a before, ~a after; ~a of ~a buffers freed, ~a by refusals\n"
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
