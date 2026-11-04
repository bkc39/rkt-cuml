#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in math/flonum flsum flvector->list)
         (only-in racket/flonum for/flvector in-flvector)
         (only-in racket/list make-list remove-duplicates)
         (only-in racket/math sqr)
         (only-in rackunit check-= check-equal?)
         (only-in raft device-vector->flvector flvector->device-vector list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/metrics-data.rkt" exact-mse exact-r2)
         (only-in "private/scores-data.rkt" vec weights-of))

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
  (define-values (r2 mse) (host-scores (device-vector->flvector y) (device-vector->flvector p)))
  (define ours-r2 (r2-score y p))
  (define ours-mse (mean-squared-error y p))
  (define r2-scale (max 1.0 (abs (- 1.0 r2))))
  (printf "X1: r2 ~a (gap ~a), error ~a (gap ~a)\n"
          ours-r2
          (/ (abs (- ours-r2 r2)) r2-scale)
          ours-mse
          (/ (abs (- ours-mse mse)) mse))
  (check-= ours-r2 r2 (* 1e-12 r2-scale) (format "r2 ~a against ~a" ours-r2 r2))
  (check-= ours-mse mse (* 1e-12 mse) (format "error ~a against ~a" ours-mse mse)))

(test-gpu "X2: five repeated weighted calls of each score give the same bits"
  (define y (flvector->device-vector (normals 3 1e4) #:dtype 'float32))
  (define p (flvector->device-vector (normals 4 1e4) #:dtype 'float32))
  (define next (uniform-source 5))
  (define w (flvector->device-vector (for/flvector #:length n ([_ (in-range n)]) (* 3.0 (next)))))
  (define labels
    (list->device-vector (for/list ([i (in-range n)])
                           (modulo i 5))
                         #:dtype 'int32))
  (define guesses
    (list->device-vector (for/list ([i (in-range n)])
                           (modulo i 7))
                         #:dtype 'int64))
  (define (repeats score)
    (for/list ([_ (in-range 5)])
      (score)))
  (for ([score (in-list (list (lambda () (r2-score y p #:sample-weight w))
                              (lambda () (mean-squared-error y p #:sample-weight w))
                              (lambda () (accuracy-score labels guesses #:sample-weight w))))])
    (define runs (repeats score))
    (check-equal? (length (remove-duplicates runs)) 1 (format "~a" runs))))

(define (r2 y p [w #f] #:dtype [dt 'float64] #:pred [pt dt])
  (r2-score (vec y dt) (vec p pt) #:sample-weight (weights-of w)))

(define (mse y p [w #f] #:dtype [dt 'float64] #:pred [pt dt])
  (mean-squared-error (vec y dt) (vec p pt) #:sample-weight (weights-of w)))

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
