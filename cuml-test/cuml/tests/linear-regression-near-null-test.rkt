#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit)
         (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" [on-matrix matrix] [on-vector vector])
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal))

(define rows 5000)
(define eps (expt 2 -12))
(define seeds 4)
(define pair-counts '(8 16 17 20 30))
(define spare-pair-counts '(17 30))
(define spare-columns 120)

(define (integers salt)
  (let loop ([s (+ 1 (* 7919 salt))]
             [i 0]
             [acc '()])
    (define next (modulo (+ (* s 1103515245) 12345) 2147483648))
    (if (= i rows)
        acc
        (loop next (add1 i) (cons (- (modulo (quotient next 65536) 23) 11) acc)))))

(define (near-pair j seed)
  (define x (integers (+ (* 100 j) seed 3)))
  (define z (integers (+ (* 100 j) seed 50003)))
  (list x (map (lambda (a b) (+ a (* eps b))) x z)))

(define (near-null-columns pairs spare seed)
  (define x1 (integers (+ 1000 seed)))
  (define x2 (integers (+ 2000 seed)))
  (append (list x1 x2 (map + x1 x2))
          (apply append
                 (for/list ([j (in-range pairs)])
                   (near-pair j seed)))
          (for/list ([j (in-range spare)])
            (integers (+ (* 100 j) seed 90007)))))

(define (targets seed)
  (for/list ([v (in-list (integers (+ seed 777)))]
             [i (in-naturals)])
    (+ v (/ (modulo (* 31 i) 17) 16))))

(define (check-refused pairs spare)
  (for ([seed (in-range seeds)])
    (define X (matrix (apply map list (near-null-columns pairs spare seed)) 'float32))
    (define y (vector (targets seed) 'float32))
    (for ([algorithm (in-list '(qr svd-qr))])
      (test-case (format "~a, ~a near pairs, ~a spare columns, seed ~a" algorithm pairs spare seed)
        (check-cuml-error 'logic
                          (collinear-refusal algorithm)
                          (lambda () (linear-regression-fit X y #:algorithm algorithm)))))))

(test-gpu "float32 'qr and 'svd-qr refuse exact collinearity beside many near pairs"
  (for ([pairs (in-list pair-counts)])
    (check-refused pairs 0)))

(test-gpu "the Ritz window holds every near-null direction beside many well-spread columns"
  (for ([pairs (in-list spare-pair-counts)])
    (check-refused pairs spare-columns)))
