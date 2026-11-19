#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/match match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/exact-algebra.rkt" dot)
         (only-in "private/linear-data.rkt"
                  gap
                  lcase
                  lcase-name
                  lcase-X
                  lcase-y
                  max-abs
                  oracle-lstsq
                  solution-coef
                  solution-intercept
                  solution-kappa
                  summed-case
                  unit-roundoff
                  [on-matrix matrix]
                  [on-vector vector])
         (only-in "private/linear-refusal-rows.rkt" collinear-refusal not-least-squares))

(define sizes '(60 1000))
(define draws 30)
(define near-sizes '(1000 5000))
(define near-draws 8)
(define near-gaps (list (expt 2 -12) (expt 2 -13)))
(define qr-share 32)
(define normal-equations-share 109)
(define prediction-bound (expt 2 -7))
(define wide-bound (expt 2 -20))

(define (fit-with algorithm c #:fit-intercept [fi #t])
  (linear-regression-fit (matrix (lcase-X c) 'float32)
                         (vector (lcase-y c) 'float32)
                         #:algorithm algorithm
                         #:fit-intercept fi))

(define (coef-of fit)
  (map inexact->exact (device-vector->list (linear-regression-result-coef fit))))

(define (relative-gap coef oracle)
  (/ (gap coef (solution-coef oracle)) (max-abs (solution-coef oracle))))

(define (prediction-gap c fit)
  (define o (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
  (define (predict b coef)
    (for/list ([row (in-list (lcase-X c))])
      (+ b (dot row coef))))
  (/ (gap (predict (inexact->exact (linear-regression-result-intercept fit)) (coef-of fit))
          (predict (solution-intercept o) (solution-coef o)))
     (max-abs (lcase-y c))))

(define (shifted-case n)
  (define c (summed-case n 0))
  (lcase (format "x3 = x1 + 3, ~a rows" n)
         (for/list ([row (in-list (lcase-X c))])
           (match-define (list a b _) row)
           (list a b (+ a 3)))
         (lcase-y c)
         #f))

(define (scrambled n salt)
  (for/list ([i (in-range n)])
    (- (modulo (+ (* i (+ 37 (* 2 salt))) (* 11 salt) (quotient (* i i) (+ 5 salt))) 23) 11)))

(define (near-pair-case n seed eps)
  (define x1 (scrambled n seed))
  (define x2 (scrambled n (+ seed 20)))
  (define x4 (scrambled n (+ seed 40)))
  (define z (scrambled n (+ seed 60)))
  (lcase (format "x3 = x1 + x2 beside x5 = x4 + ~a z, ~a rows, seed ~a" eps n seed)
         (for/list ([a (in-list x1)]
                    [b (in-list x2)]
                    [c (in-list x4)]
                    [e (in-list z)])
           (list a b (+ a b) c (+ c (* eps e))))
         (for/list ([v (in-list (scrambled n (+ seed 80)))]
                    [i (in-naturals)])
           (+ v (/ (modulo (* 31 i) 17) 16)))
         #f))

(test-gpu "float32 'qr and 'svd-qr refuse exactly collinear X"
  (for* ([algorithm (in-list '(qr svd-qr))]
         [n (in-list sizes)]
         [seed (in-range draws)])
    (define c (summed-case n seed))
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (check-cuml-error 'logic (collinear-refusal algorithm) (lambda () (fit-with algorithm c))))))

(test-gpu "the Gram check sees exact collinearity beside a second near-null direction"
  (for* ([algorithm (in-list '(qr svd-qr))]
         [n (in-list near-sizes)]
         [eps (in-list near-gaps)]
         [seed (in-range near-draws)])
    (define c (near-pair-case n seed eps))
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (check-cuml-error 'logic (collinear-refusal algorithm) (lambda () (fit-with algorithm c))))))

(test-gpu "float32 'auto, 'eig, 'svd and 'svd-jacobi fit exactly collinear X or refuse it as not least squares"
  (for* ([algorithm (in-list '(auto eig svd svd-jacobi))]
         [n (in-list sizes)]
         [seed (in-range draws)])
    (define c (summed-case n seed))
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (match (with-handlers ([exn:fail? exn-message])
               (fit-with algorithm c))
        [(? string? message) (check-true (regexp-match? not-least-squares message) message)]
        [fit (check-true (< (prediction-gap c fit) prediction-bound))]))))

(test-gpu "float32 'qr and 'svd-qr fit wide X by the Jacobi SVD without the Gram check"
  (define c (lcase "4 by 5" '((1 0 2 1 3) (0 3 1 4 1) (4 1 0 2 2) (2 2 5 0 1)) '(1 2 3 4) #f))
  (for ([algorithm (in-list '(qr svd-qr))])
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (check-true (< (prediction-gap c (fit-with algorithm c)) wide-bound)))))

(test-gpu "columns collinear only after centring are refused with an intercept and fit without"
  (for* ([algorithm (in-list '(qr svd-qr))]
         [n (in-list sizes)])
    (define c (shifted-case n))
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (check-cuml-error 'logic (collinear-refusal algorithm) (lambda () (fit-with algorithm c)))
      (define oracle (oracle-lstsq (lcase-X c) (lcase-y c) #f #f))
      (check-true (<= (relative-gap (coef-of (fit-with algorithm c #:fit-intercept #f)) oracle)
                      (* qr-share (unit-roundoff 'float32) (solution-kappa oracle)))))))

(define share-rows
  `((auto 1/4 ,normal-equations-share) (eig 1/4 ,normal-equations-share)
                                       (svd 1/128 ,qr-share)
                                       (svd-jacobi 1/128 ,qr-share)
                                       (qr 1/128 ,qr-share)
                                       (svd-qr 1/128 ,qr-share)))

(test-gpu "every float32 algorithm fits nearly collinear X within its share of u kappa"
  (for* ([row (in-list share-rows)]
         [n (in-list sizes)]
         [seed (in-range draws)])
    (match-define (list algorithm eps share) row)
    (define c (summed-case n seed eps))
    (define oracle (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
    (test-case (format "~a ~a" algorithm (lcase-name c))
      (check-true (<= (relative-gap (coef-of (fit-with algorithm c)) oracle)
                      (* share (unit-roundoff 'float32) (solution-kappa oracle)))))))
