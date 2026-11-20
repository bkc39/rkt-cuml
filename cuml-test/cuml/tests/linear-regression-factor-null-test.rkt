#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit)
         (only-in racket/fixnum fxpopcount)
         (only-in racket/flonum flvector-ref flvector-set! make-flvector)
         (only-in racket/list split-at)
         (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft device-matrix)
         (only-in raft/private/array write-array!)
         (only-in raft/private/pack pack-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" [on-vector vector])
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" integer-draws wobble [on-matrix matrix])
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal))

(define structured-nulls
  (list '(dummies-beside-ones 23379 #f)
        '(dummies-beside-ones 57115 #f)
        '(signs-summing-to-zero 4596 #f)
        '(signs-summing-to-zero 4596 #t)
        '(signs-summing-to-zero 4818 #f)
        '(signs-summing-to-zero 4818 #t)
        '(hadamard-pair 0 #f)
        '(hadamard-pair 0 #t)
        '(hadamard-pair 10 #f)
        '(hadamard-pair 10 #t)))

(define sign-patterns
  #(#(1 1 -1 -1) #(1 -1 1 -1) #(1 -1 -1 1) #(-1 1 1 -1) #(-1 1 -1 1) #(-1 -1 1 1)))

(define hadamard-rows 1024)
(define hadamard-loading 65536.0)

(define (hadamard i j)
  (if (even? (fxpopcount (bitwise-and i j))) 1.0 -1.0))

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
  (values (matrix (for/list ([i (in-range n)])
                    (for/list ([j (in-range d)])
                      (structured-entry kind size i j)))
                  'float32)
          (vector (for/list ([i (in-range n)])
                    (wobble i))
                  'float32)))

(define rows 8000)
(define columns 2000)
(define loading 10.0)
(define seeds 4)

(define (factor-columns seed)
  (define-values (factors noise) (split-at (integer-draws seed (* rows (add1 columns))) rows))
  (define loadings
    (for/vector #:length rows
                ([f (in-list factors)])
      (* loading (exact->inexact f))))
  (define xs (make-flvector (* rows columns)))
  (for ([e (in-list noise)]
        [k (in-naturals)])
    (flvector-set! xs k (+ (vector-ref loadings (quotient k columns)) e)))
  (for ([at (in-range 0 (* rows columns) columns)])
    (flvector-set! xs
                   (+ at 2)
                   (- (+ (flvector-ref xs at) (flvector-ref xs (+ at 1)))
                      (flvector-ref xs (+ at 3)))))
  (define X (device-matrix rows columns #:dtype 'float32))
  (void (write-array! 'factor-columns X (pack-vector 'factor-columns 'float32 (* rows columns) xs)))
  X)

(define (targets seed)
  (vector (for/list ([v (in-list (integer-draws (+ seed 777) rows))]
                     [i (in-naturals)])
            (+ v (wobble i)))
          'float32))

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
    (match-define (list kind size intercept) row)
    (define-values (X y) (structured-design kind size))
    (for ([algorithm (in-list '(qr svd-qr))])
      (test-case (format "~a, ~a ~a, intercept ~a" algorithm kind size intercept)
        (check-cuml-error
         'logic
         (collinear-refusal algorithm)
         (lambda () (linear-regression-fit X y #:algorithm algorithm #:fit-intercept intercept)))))))
