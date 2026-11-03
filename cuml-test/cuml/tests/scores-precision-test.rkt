#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in math/flonum flsum)
         (only-in racket/flonum for/flvector in-flvector)
         (only-in racket/list make-list remove-duplicates)
         (only-in racket/math sqr)
         (only-in rackunit check-equal? check-true)
         (only-in raft device-vector->flvector flvector->device-vector list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source))

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
  (define ys
    (for/list ([a (in-flvector y)])
      a))
  (define ps
    (for/list ([b (in-flvector p)])
      b))
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
  (define r2-gap (/ (abs (- ours-r2 r2)) (max 1.0 (abs (- 1.0 r2)))))
  (define mse-gap (/ (abs (- ours-mse mse)) mse))
  (printf "X1: r2 ~a (gap ~a), error ~a (gap ~a)\n" ours-r2 r2-gap ours-mse mse-gap)
  (check-true (<= r2-gap 1e-12) (format "r2 ~a against ~a" ours-r2 r2))
  (check-true (<= mse-gap 1e-12) (format "error ~a against ~a" ours-mse mse)))

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
