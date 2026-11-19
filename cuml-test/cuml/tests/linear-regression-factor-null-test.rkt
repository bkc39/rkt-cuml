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

(define structured-nulls
  '((dummies-beside-ones 23379 #f) (dummies-beside-ones 57115 #f)
                                   (signs-summing-to-zero 4596 #f)
                                   (signs-summing-to-zero 4596 #t)
                                   (signs-summing-to-zero 4818 #f)
                                   (signs-summing-to-zero 4818 #t)
                                   (hadamard-pair 0 #f)
                                   (hadamard-pair 0 #t)
                                   (hadamard-pair 10 #f)
                                   (hadamard-pair 10 #t)))

(define sign-patterns
  #(#(1 1 -1 -1) #(1 -1 1 -1) #(1 -1 -1 1) #(-1 1 1 -1) #(-1 1 -1 1) #(-1 -1 1 1)))

(define hadamard-rows 1024)
(define hadamard-loading 65536.0)

(define (hadamard i j)
  (if (even? (popcount (bitwise-and i j))) 1.0 -1.0))

(define (popcount v)
  (for/sum ([b (in-range (integer-length v))]) (if (bitwise-bit-set? v b) 1 0)))

(define (structured-entry kind size i j)
  (case kind
    [(dummies-beside-ones) (if (or (= j 2) (= j (modulo i 2))) 1.0 0.0)]
    [(signs-summing-to-zero) (exact->inexact (vector-ref (vector-ref sign-patterns (modulo i 6)) j))]
    [(hadamard-pair)
     (* (+ (hadamard i (add1 j)) (* hadamard-loading (hadamard i (sub1 hadamard-rows))))
        (expt 2.0 (* size j)))]))

(define (structured-shape kind size)
  (case kind
    [(dummies-beside-ones) (values size 3)]
    [(signs-summing-to-zero) (values size 4)]
    [(hadamard-pair) (values hadamard-rows 2)]))

(define (structured-design kind size)
  (define-values (n d) (structured-shape kind size))
  (define xs (make-flvector (* n d)))
  (for* ([i (in-range n)]
         [j (in-range d)])
    (flvector-set! xs (+ (* i d) j) (structured-entry kind size i j)))
  (define X (device-matrix n d #:dtype 'float32))
  (void (write-array! 'structured-design X (pack-vector 'structured-design 'float32 (* n d) xs)))
  (define y (make-flvector n))
  (for ([i (in-range n)])
    (flvector-set! y i (exact->inexact (/ (modulo (* 31 i) 17) 16))))
  (values X (flvector->device-vector y #:dtype 'float32)))

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

(test-gpu "float32 'qr and 'svd-qr refuse structured null vectors the float32 Gram rounds away"
  (for ([row (in-list structured-nulls)])
    (define-values (kind size intercept) (apply values row))
    (define-values (X y) (structured-design kind size))
    (for ([algorithm (in-list '(qr svd-qr))])
      (test-case (format "~a, ~a ~a, intercept ~a" algorithm kind size intercept)
        (check-cuml-error
         'logic
         (collinear-refusal algorithm)
         (lambda () (linear-regression-fit X y #:algorithm algorithm #:fit-intercept intercept)))))))
