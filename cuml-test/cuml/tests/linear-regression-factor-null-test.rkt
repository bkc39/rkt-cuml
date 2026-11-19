#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit)
         (only-in racket/flonum flvector-ref flvector-set! make-flvector)
         (only-in rackunit test-case)
         (only-in raft device-matrix flvector->device-vector)
         (only-in raft/private/array write-array!)
         (only-in raft/private/pack pack-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal))

(define rows 8000)
(define columns 2000)
(define loading 10.0)
(define seeds 4)

(define (draws salt count)
  (define v (make-flvector count))
  (let loop ([s (+ 1 (* 7919 salt))]
             [i 0])
    (when (< i count)
      (define next (modulo (+ (* s 1103515245) 12345) 2147483648))
      (flvector-set! v i (exact->inexact (- (modulo (quotient next 65536) 23) 11)))
      (loop next (add1 i))))
  v)

(define (factor-columns seed)
  (define noise (draws seed (* rows (add1 columns))))
  (define xs (make-flvector (* rows columns)))
  (for* ([i (in-range rows)]
         [j (in-range columns)])
    (flvector-set! xs
                   (+ (* i columns) j)
                   (+ (* loading (flvector-ref noise i))
                      (flvector-ref noise (+ rows (* i columns) j)))))
  (for ([i (in-range rows)])
    (define at (* i columns))
    (flvector-set! xs
                   (+ at 2)
                   (- (+ (flvector-ref xs at) (flvector-ref xs (+ at 1)))
                      (flvector-ref xs (+ at 3)))))
  (define X (device-matrix rows columns #:dtype 'float32))
  (void (write-array! 'factor-columns X (pack-vector 'factor-columns 'float32 (* rows columns) xs)))
  X)

(define (targets seed)
  (define v (draws (+ seed 777) rows))
  (for ([i (in-range rows)])
    (flvector-set! v i (+ (flvector-ref v i) (/ (modulo (* 31 i) 17) 16.0))))
  (flvector->device-vector v #:dtype 'float32))

(test-gpu "float32 'qr and 'svd-qr refuse exact collinearity among columns sharing one strong factor"
  (for ([seed (in-range seeds)])
    (define X (factor-columns seed))
    (define y (targets seed))
    (for ([algorithm (in-list '(qr svd-qr))])
      (test-case (format "~a, ~ax~a, loading ~a, seed ~a" algorithm rows columns loading seed)
        (check-cuml-error 'logic
                          (collinear-refusal algorithm)
                          (lambda () (linear-regression-fit X y #:algorithm algorithm)))))))
