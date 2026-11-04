#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in math/flonum flsum flvector->list)
         (only-in racket/flonum for/flvector in-flvector)
         (only-in racket/list make-list remove-duplicates)
         (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check-= check-equal?)
         (only-in raft device-vector->flvector flvector->device-vector list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/metrics-data.rkt" exact-mse exact-r2)
         (only-in "private/scores-data.rkt" mse r2))

(define n (expt 10 6))

(test-gpu "A5: accuracy over 2^24 + 3 samples with one wrong is the exact count over n"
  (define size (+ (expt 2 24) 3))
  (define truth (list->device-vector (make-list size 0) #:dtype 'int32))
  (define found (list->device-vector (cons 1 (make-list (sub1 size) 0)) #:dtype 'int32))
  (check-equal? (accuracy-score truth found) 0.9999999403953659))

(define (normals seed shift)
  (define next (normal-source seed))
  (for/flvector #:length n ([_ (in-range n)]) (+ shift (next))))

(define (host-scores y p)
  (define ys (flvector->list y))
  (define ps (flvector->list p))
  (define count (exact->inexact (length ys)))
  (define centre (/ (flsum ys) count))
  (define spread (flsum (map (lambda (a) (sqr (- a centre))) ys)))
  (define residual (flsum (map (lambda (a b) (sqr (- a b))) ys ps)))
  (values (- 1.0 (/ residual spread)) (/ residual count)))

(test-gpu "X1: float32 r2 and error are within 1e-12 of a float64 two-pass sum over the device's values"
  (define y (flvector->device-vector (normals 1 1e4) #:dtype 'float32))
  (define offsets (normals 2 0.0))
  (define p
    (flvector->device-vector
     (for/flvector #:length n
                   ([a (in-flvector (device-vector->flvector y))] [e (in-flvector offsets)])
                   (+ a e))
     #:dtype 'float32))
  (define-values (host-r2 host-mse)
    (host-scores (device-vector->flvector y) (device-vector->flvector p)))
  (define ours-r2 (r2-score y p))
  (define ours-mse (mean-squared-error y p))
  (define r2-scale (max 1.0 (abs (- 1.0 host-r2))))
  (printf "X1: r2 ~a (gap ~a), error ~a (gap ~a)\n"
          ours-r2
          (/ (abs (- ours-r2 host-r2)) r2-scale)
          ours-mse
          (/ (abs (- ours-mse host-mse)) host-mse))
  (check-= ours-r2 host-r2 (* 1e-12 r2-scale) (format "r2 ~a against ~a" ours-r2 host-r2))
  (check-= ours-mse host-mse (* 1e-12 host-mse) (format "error ~a against ~a" ours-mse host-mse)))

(test-gpu "X2: five repeated weighted calls of each score give the same bits"
  (define y (flvector->device-vector (normals 3 1e4) #:dtype 'float32))
  (define p (flvector->device-vector (normals 4 1e4) #:dtype 'float32))
  (define next (uniform-source 5))
  (define w (flvector->device-vector (for/flvector #:length n ([_ (in-range n)]) (* 3.0 (next)))))
  (define (cycled period dt)
    (list->device-vector (for/list ([_ (in-range n)]
                                    [x (in-cycle (in-range period))])
                           x)
                         #:dtype dt))
  (define labels (cycled 5 'int32))
  (define guesses (cycled 7 'int64))
  (define (repeats score)
    (for/list ([_ (in-range 5)])
      (score)))
  (for ([score (in-list (list (lambda () (r2-score y p #:sample-weight w))
                              (lambda () (mean-squared-error y p #:sample-weight w))
                              (lambda () (accuracy-score labels guesses #:sample-weight w))))])
    (define runs (repeats score))
    (check-equal? (length (remove-duplicates runs)) 1 (format "~a" runs))))

(test-gpu "a small residual beside a large value keeps its bits"
  (check-equal? (mse '(1e200 1.0) '(1e200 2.0)) 0.5)
  (check-equal? (mse (list (expt 2.0 540) 1.0) (list (expt 2.0 540) 2.0)) 0.5)
  (check-equal? (mse (list (expt 2.0 127) 1.0) (list (expt 2.0 127) 2.0) #:dtype 'float32) 0.5)
  (define exact (exact-mse '(1e160 1.0) '(1e160 1.1)))
  (check-= (inexact->exact (mse '(1e160 1.0) '(1e160 1.1))) exact (* 1e-14 exact)))

(test-gpu "a large value behind a zero weight scores as the call without its sample"
  (define w '(0.0 1.0 1.0))
  (check-equal? (r2 (list (expt 2.0 1000) 1.0 2.0) (list (expt 2.0 1000) 1.0 3.0) w) -1.0)
  (check-equal? (r2 '(1.0 2.0) '(1.0 3.0)) -1.0)
  (check-equal? (mse (list (expt 2.0 1000) 1.0 2.0) (list (expt 2.0 1000) 1.0 3.0) w) 0.5)
  (check-equal? (mse '(1.0 2.0) '(1.0 3.0)) 0.5)
  (check-equal? (r2 '(0.0 1.0 2.0) '(1e300 1.0 3.0) w #:dtype 'float32 #:pred 'float64) -1.0)
  (check-equal? (mse '(0.0 1.0 2.0) '(1e300 1.0 3.0) w #:dtype 'float32 #:pred 'float64) 0.5)
  (check-equal? (r2 (list (expt 2.0 127) 1.0 2.0) (list (expt 2.0 127) 1.0 3.0) w #:dtype 'float32)
                -1.0)
  (define exact (exact-r2 '(1.0 2.0) '(1.1 2.0)))
  (check-= (inexact->exact (r2 '(1e160 1.0 2.0) '(1e160 1.1 2.0) w))
           exact
           (* 1e-14 (abs (- 1 exact)))))

(test-gpu "a weight that scales to zero against the largest drops its sample"
  (define w (list (expt 2.0 1023) (expt 2.0 -1000)))
  (check-equal? (r2 '(5.0 7.0) '(6.0 8.0) w) (r2 '(5.0) '(6.0)))
  (check-equal? (r2 '(5.0 7.0) '(6.0 8.0) w) 0.0)
  (check-equal? (r2 '(5.0 7.0) '(5.0 8.0) w) 1.0)
  (check-equal? (mse '(5.0 7.0) '(6.0 8.0) w) 1.0))

(test-gpu "a small residual beside a large zero residual keeps its bits in the error"
  (for ([c (in-list (list (list '(1e300 0.0) '(1e300 1e-30) #f)
                          (list '(-1e300 0.0) '(-1e300 1e-30) '(3.0 1.0))
                          (list '((1e300 0.0)) '((1e300 1e-30)) #f)))])
    (match-define (list y p w) c)
    (check-equal? (mse y p w) (exact->inexact (exact-mse y p w)) (format "~a" c)))
  (for ([c (in-list (list (list '(-1e300 0.0) '(-1e300 1e-30) '(1.0 3.0))
                          (list '(1e300 0.0) '(1e300 1e-12) #f)))])
    (match-define (list y p w) c)
    (define exact (exact-mse y p w))
    (check-= (inexact->exact (mse y p w)) exact (* 1e-14 exact) (format "~a" c))))

(test-gpu "an r2 whose rescaled ratio overflows is finite when the score is"
  (define y '(0.0 1.0))
  (define p (list (expt 2.0 -10) 1.0))
  (define w (list (expt 2.0 1000) (expt 2.0 -30)))
  (check-equal? (r2 y p w) (exact->inexact (exact-r2 y p w))))

(test-gpu "an r2 whose deviations all flush to zero under the value scaling is -inf.0"
  (check-equal? (r2 (list (expt 2.0 -60) (expt 2.0 -59)) (list (expt 2.0 1023) 0.0)) -inf.0))

(test-gpu "weights below 2^-1022 of the largest leave the score within 2^(e-1075)/w"
  (define y '(0.0 1.0 -1.0))
  (define p '(0.0 2.0 -1.0))
  (define w (list (expt 2.0 1000) (* 1.3 (expt 2.0 -60)) (expt 2.0 -60)))
  (define exact (exact-r2 y p w))
  (check-= (inexact->exact (r2 y p w)) exact (* (expt 2 (- 1000 1075 -60)) (abs exact))))

(test-gpu "a zero-weight sample whose scaled value overflows changes nothing"
  (define y '(1.7e308 1e-150 2e-150))
  (define p '(-1.7e308 1e-150 3e-150))
  (define w '(0.0 1.0 1.0))
  (check-equal? (r2 y p w) (exact->inexact (exact-r2 y p w)))
  (check-equal? (mse y p w) (exact->inexact (exact-mse y p w)))
  (define columns (map list y))
  (check-equal? (mse columns (map list p) w) (exact->inexact (exact-mse columns (map list p) w)))
  (define y-rows '((1.7e308 -1.7e308) (1e-150 2e-150)))
  (define p-rows '((-1.7e308 1.7e308) (1e-150 3e-150)))
  (check-equal? (mse y-rows p-rows '(0.0 1.0)) (exact->inexact (exact-mse y-rows p-rows '(0.0 1.0)))))

(define (light-beside-heavy count heavy)
  (values (cons 0.0 heavy)
          (cons (expt 2.0 500) heavy)
          (cons (* 1.3 (expt 2.0 -1022)) (make-list count 1.0))))

(define (alternating count c)
  (for/list ([_ (in-range count)]
             [sign (in-cycle '(1.0 -1.0))])
    (* sign c)))

(test-gpu "a light normal weight on the largest residual leaves the final ratio's bits"
  (for ([count (in-list '(1024 65536))])
    (define-values (y p w) (light-beside-heavy count (alternating count (* 1.9 (expt 2.0 -20)))))
    (define exact (exact-r2 y p w))
    (check-= (inexact->exact (r2 y p w)) exact (* 1e-14 (abs (- 1 exact))) (format "r2 n = ~a" count))
    (define-values (ym pm wm) (light-beside-heavy count (make-list count 1.0)))
    (check-equal? (mse ym pm wm)
                  (exact->inexact (exact-mse ym pm wm))
                  (format "error n = ~a" count))))
