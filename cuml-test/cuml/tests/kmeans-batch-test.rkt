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
         (only-in racket/list append* first second third)
         (only-in racket/match == match match-define)
         (only-in rackunit
                  check-equal?
                  check-exn
                  check-false
                  check-not-eq?
                  check-regexp-match
                  check-true
                  exn:test:check?
                  test-case)
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

(define (spread actual expected)
  (for/list ([a (in-list actual)]
             [b (in-list expected)])
    (list (deviation a b) (equal? a b))))

(define (check-close-all! name actual expected tol)
  (define far
    (for/first ([a (in-list actual)]
                [b (in-list expected)]
                #:unless (close? a b tol))
      (list a b)))
  (check-false far (format "~a: ~a" name far)))

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

(define (numbers r)
  (match-define (kmeans-result C _ inertia _) r)
  (cons inertia (append* (host C))))

(define (partition r)
  (match-define (kmeans-result _ labels _ n-iter) r)
  (list (host-vector labels) n-iter))

(define (same-fit? actual expected dt)
  (and (equal? (partition actual) (partition expected))
       (for/and ([a (in-list (numbers actual))]
                 [b (in-list (numbers expected))])
         (close? a b (tolerance dt)))))

(define escalation 10)

(define (miss agrees? thunk)
  (if (agrees? (thunk)) 0 1))

(define (interleaved-misses agrees? rerun refit)
  (for/fold ([rerun-misses 0]
             [refit-misses 0]
             #:result (list rerun-misses refit-misses))
            ([_ (in-range escalation)])
    (values (+ rerun-misses (miss agrees? rerun)) (+ refit-misses (miss agrees? refit)))))

(define (batch-verdict agrees? first rerun refit)
  (match (and (not (agrees? first)) (interleaved-misses agrees? rerun refit))
    [#f 'agrees]
    [(list (== escalation) (or 0 1)) 'departed]
    [_ 'unstable]))

(define (check-batch-fit! name agrees? first rerun refit)
  (define verdict (batch-verdict agrees? first rerun refit))
  (check-not-eq?
   verdict
   'departed
   (format "~a: the batch disagreed in all ~a reruns, the default in at most 1 refit between them"
           name
           escalation))
  verdict)

(define (check-same-apply name dt C X w batch)
  (define (applied b)
    (parameterize ([current-batch-samples b])
      (list (host-vector (kmeans-predict C X))
            (cons (kmeans-score C X #:sample-weight w) (append* (host (kmeans-transform C X)))))))
  (match-define (list labels measured) (applied batch))
  (match-define (list labels0 measured0) (applied 32768))
  (check-equal? labels labels0 (format "~a: predict" name))
  (check-close-all! (format "~a: score or transform" name) measured measured0 (tolerance dt))
  (spread measured measured0))

(define (replaying outcomes)
  (define-values (_more? next) (sequence-generate (in-cycle outcomes)))
  next)

(test-case "H13: the escalation fails a batch that departs and passes a nondeterministic fit"
  (define (verdict first rerun refit)
    (with-handlers ([exn:test:check? (lambda (_) 'failed)])
      (check-batch-fit! "stub" values first rerun refit)))
  (check-equal? (verdict #t (replaying '(#f)) (replaying '(#t))) 'agrees)
  (check-equal? (verdict #f (replaying '(#f)) (replaying '(#t))) 'failed)
  (check-equal? (verdict #f (replaying '(#f)) (replaying '(#t #t #t #t #t #t #t #t #t #f))) 'failed)
  (check-equal? (verdict #f (replaying '(#f #f #f #f #f #f #f #f #f #t)) (replaying '(#t))) 'unstable)
  (check-equal? (verdict #f (replaying '(#t #f)) (replaying '(#t #f))) 'unstable)
  (check-equal? (verdict #f (replaying '(#f #f #t)) (replaying '(#t))) 'unstable)
  (check-equal? (verdict #f (replaying '(#f)) (replaying '(#t #t #t #t #f))) 'unstable))

(define (burst calls)
  (define count 0)
  (lambda ()
    (set! count (add1 count))
    (> count calls)))

(test-case "M-2: a burst of disagreement across the reruns and the refits is no departure"
  (define calls (burst escalation))
  (check-equal? (batch-verdict values #f calls calls) 'unstable))

(test-gpu "H13: the batch size changes no label and no n-iter"
  (define fits
    (for*/list ([dt (in-list dtypes)]
                [X (in-value (blob-data rows 4 16 11 dt))]
                [init (in-list inits)]
                [n-init (in-list '(1 3))]
                [w (in-list (list #f (vec (cyclic-weights rows) dt)))]
                [reference (in-value (fitted X dt init n-init w 32768))]
                [batch (in-list (list 7 100 (sub1 rows) rows))])
      (define name (format "~a ~a n-init ~a~a batch ~a" dt init n-init (if w " weighted" "") batch))
      (define actual (fitted X dt init n-init w batch))
      (define verdict
        (check-batch-fit! name
                          (lambda (r) (same-fit? r reference dt))
                          actual
                          (lambda () (fitted X dt init n-init w batch))
                          (lambda () (fitted X dt init n-init w 32768))))
      (define applied (check-same-apply name dt (kmeans-result-centroids reference) X w batch))
      (list name
            verdict
            (match verdict
              ['agrees (append (spread (numbers actual) (numbers reference)) applied)]
              [_ applied]))))
  (define grids
    (for*/list ([dt (in-list dtypes)]
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
      (check-same-apply name dt (kmeans-result-centroids reference) X #f batch)))
  (define spreads (append* (append (map third fits) grids)))
  (define (named verdict)
    (for/list ([fit (in-list fits)]
               #:when (eq? (second fit) verdict))
      (first fit)))
  (printf
   "H13: largest relative centroid, inertia, distance or score deviation ~a; bit-identical: ~a\n"
   (apply max 0.0 (map first spreads))
   (andmap second spreads))
  (printf "H13: fits that also change between runs at the default batch: ~a\n" (named 'unstable))
  (check-equal? (named 'departed) '()))

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
  (second (sort times <)))

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
              (format "built inside ~a ms, called inside ~a ms" outside-call inside-call))
  (define fitted-with-setting (fit (kmeans #:init C #:max-samples-per-batch 1) C))
  (define setting-call (median-time (lambda () (transform fitted-with-setting X))))
  (check-true (< (* 20 outside-call) setting-call)
              (format "default ~a ms, the setting at batch 1 ~a ms" outside-call setting-call)))

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
