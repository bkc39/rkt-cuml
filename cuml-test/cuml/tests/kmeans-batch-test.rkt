#lang racket/base

(require (only-in cuml fit kmeans kmeans-model transform)
         (only-in cuml/cluster
                  current-batch-samples
                  kmeans-fit
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-result-inertia
                  kmeans-score
                  kmeans-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-regexp-match check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/kmeans-data.rkt"
                  blob-data
                  close?
                  cyclic-weights
                  dtypes
                  first-rows
                  grid64
                  grid64-start
                  host
                  host-vector
                  mat
                  tolerance
                  vec))

(define rows 4096)

(define inits '(k-means++ scalable-k-means++ random matrix))

(define (deviation a b)
  (/ (abs (- a b)) (max 1.0 (abs b))))

(struct tally (largest bit-identical?) #:mutable)

(define (tallied! t actual expected)
  (define d (deviation actual expected))
  (set-tally-largest! t (max (tally-largest t) d))
  (set-tally-bit-identical?! t (and (tally-bit-identical? t) (equal? actual expected))))

(define (check-close! t actual expected tol name)
  (tallied! t actual expected)
  (check-true (close? actual expected tol) (format "~a: ~a against ~a" name actual expected)))

(define (check-rows! t actual expected tol name)
  (for* ([(actual-row expected-row) (in-parallel actual expected)]
         [(a b) (in-parallel actual-row expected-row)])
    (check-close! t a b tol name)))

(define (fitted X dt init n-init w batch)
  (define start (and (eq? init 'matrix) (first-rows X 16 dt)))
  (parameterize ([current-batch-samples batch])
    (kmeans-fit X
                #:n-clusters 16
                #:init (or start init)
                #:n-init n-init
                #:random-state 3
                #:sample-weight w)))

(define (outcome r)
  (match-define (kmeans-result C labels inertia n-iter) r)
  (list (host C) (host-vector labels) inertia n-iter))

(define (partition r)
  (match-define (kmeans-result _ labels _ n-iter) r)
  (list (host-vector labels) n-iter))

(define (check-same-fit! t name dt actual expected)
  (match-define (kmeans-result C labels inertia n-iter) actual)
  (match-define (kmeans-result C0 labels0 inertia0 n-iter0) expected)
  (check-equal? (host-vector labels) (host-vector labels0) (format "~a: labels" name))
  (check-equal? n-iter n-iter0 (format "~a: n-iter" name))
  (check-rows! t (host C) (host C0) (tolerance dt) (format "~a: centroids" name))
  (check-close! t inertia inertia0 (tolerance dt) (format "~a: inertia" name)))

(define (agrees? actual expected dt)
  (match-define (kmeans-result C _ inertia _) actual)
  (match-define (kmeans-result C0 _ inertia0 _) expected)
  (and (equal? (partition actual) (partition expected))
       (close? inertia inertia0 (tolerance dt))
       (for*/and ([(row row0) (in-parallel (host C) (host C0))]
                  [(a b) (in-parallel row row0)])
         (close? a b (tolerance dt)))))

(define (record! box name)
  (set-box! box (cons name (unbox box))))

(define (check-batch-fit! t unstable departed name dt refit rerun expected)
  (define actual (rerun))
  (cond
    [(agrees? actual expected dt) (check-same-fit! t name dt actual expected)]
    [(for/or ([_ (in-range 3)])
       (or (not (agrees? (refit) expected dt)) (agrees? (rerun) expected dt)))
     (record! unstable name)]
    [else (record! departed name)]))

(define (check-same-apply! t name dt C X w batch)
  (define (applied b)
    (parameterize ([current-batch-samples b])
      (list (host-vector (kmeans-predict C X))
            (host (kmeans-transform C X))
            (kmeans-score C X #:sample-weight w))))
  (match-define (list labels distances score) (applied batch))
  (match-define (list labels0 distances0 score0) (applied 32768))
  (check-equal? labels labels0 (format "~a: predict" name))
  (check-rows! t distances distances0 (tolerance dt) (format "~a: transform" name))
  (check-close! t score score0 (tolerance dt) (format "~a: score" name)))

(test-gpu "H13: the batch size changes no label and no n-iter"
  (define t (tally 0.0 #t))
  (define unstable (box '()))
  (define departed (box '()))
  (for* ([dt (in-list dtypes)]
         [X (in-value (blob-data rows 4 16 11 dt))]
         [init (in-list inits)]
         [n-init (in-list '(1 3))]
         [w (in-list (list #f (vec (cyclic-weights rows) dt)))])
    (define reference (fitted X dt init n-init w 32768))
    (define C (kmeans-result-centroids reference))
    (for ([batch (in-list (list 7 100 (sub1 rows) rows))])
      (define name (format "~a ~a n-init ~a~a batch ~a" dt init n-init (if w " weighted" "") batch))
      (check-batch-fit! t
                        unstable
                        departed
                        name
                        dt
                        (lambda () (fitted X dt init n-init w 32768))
                        (lambda () (fitted X dt init n-init w batch))
                        reference)
      (check-same-apply! t name dt C X w batch)))
  (for* ([dt (in-list dtypes)]
         [batch (in-list '(1 7 63 64))])
    (define X (mat grid64 dt))
    (define name (format "grid64 ~a batch ~a" dt batch))
    (define reference (fitted X dt 'random 1 #f 32768))
    (define (started)
      (outcome (kmeans-fit X #:init (mat grid64-start dt))))
    (check-equal? (outcome (fitted X dt 'random 1 #f batch)) (outcome reference) name)
    (check-equal? (parameterize ([current-batch-samples batch])
                    (started))
                  (started)
                  (format "~a, start" name))
    (check-same-apply! t name dt (kmeans-result-centroids reference) X #f batch))
  (printf
   "H13: largest relative centroid, inertia, distance or score deviation ~a; bit-identical: ~a\n"
   (tally-largest t)
   (tally-bit-identical? t))
  (printf "H13: fits that also change between runs at the default batch: ~a\n"
          (reverse (unbox unstable)))
  (printf "H13: fits that departed at the batch size in 4 runs while 3 at the default agreed: ~a\n"
          (reverse (unbox departed))))

(test-gpu "H13: a parameterize, the keyword and the estimator's setting choose the same batch"
  (for ([dt (in-list dtypes)])
    (define X (mat grid64 dt))
    (define by-parameter
      (parameterize ([current-batch-samples 7])
        (kmeans-fit X #:n-clusters 4 #:init 'random #:random-state 2)))
    (define by-keyword
      (kmeans-fit X #:n-clusters 4 #:init 'random #:random-state 2 #:max-samples-per-batch 7))
    (define by-setting
      (kmeans-model
       (fit (kmeans #:n-clusters 4 #:init 'random #:random-state 2 #:max-samples-per-batch 7) X)))
    (check-equal? (outcome by-keyword) (outcome by-parameter))
    (check-equal? (outcome by-setting) (outcome by-parameter))))

(define (median-time thunk)
  (thunk)
  (define times
    (for/list ([_ (in-range 3)])
      (define start (current-inexact-milliseconds))
      (thunk)
      (- (current-inexact-milliseconds) start)))
  (list-ref (sort times <) 1))

(test-gpu "H14: the batch size is read when the procedure or the estimator is called"
  (define X (blob-data 65536 8 64 7 'float32))
  (define C (first-rows X 64 'float32))
  (define (transformed)
    (kmeans-transform C X))
  (define fast (median-time transformed))
  (define slow
    (parameterize ([current-batch-samples 1])
      (median-time transformed)))
  (printf "H14: kmeans-transform ~a ms by default, ~a ms at batch 1\n" fast slow)
  (check-true (>= slow (* 20 fast)) (format "~a ms is not 20 times ~a ms" slow fast))
  (define inside
    (parameterize ([current-batch-samples 1])
      (kmeans #:init C)))
  (define fitted-inside (fit inside C))
  (define fitted-outside (fit (kmeans #:init C) C))
  (define outside-call (median-time (lambda () (transform fitted-inside X))))
  (define inside-call
    (parameterize ([current-batch-samples 1])
      (median-time (lambda () (transform fitted-outside X)))))
  (check-true (< (* 20 outside-call) inside-call)
              (format "built inside ~a ms, called inside ~a ms" outside-call inside-call)))

(test-case "H15: current-batch-samples takes a 32-bit sample count"
  (check-equal? (current-batch-samples) 32768)
  (for ([v (in-list (list -1 2.5 (expt 2 31) 'x))])
    (define e (raised (lambda () (current-batch-samples v))))
    (check-true (exn:fail:contract? e) (format "~v" v))
    (check-regexp-match
     #rx"^current-batch-samples: contract violation\n  expected: \\(integer-in 0 2147483647\\)"
     (exn-message e)))
  (check-equal? (parameterize ([current-batch-samples 0])
                  (current-batch-samples))
                0)
  (check-equal? (parameterize ([current-batch-samples 2147483647])
                  (current-batch-samples))
                2147483647)
  (check-exn exn:fail:contract?
             (lambda ()
               (parameterize ([current-batch-samples -1])
                 0))))
