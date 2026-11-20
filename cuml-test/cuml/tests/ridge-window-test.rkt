#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-result-coef)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" columns-of on-vector unit-roundoff)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/exact-algebra.rkt" equilibrated-condition map-diagonal)
         (only-in "private/linear-data.rkt"
                  gap
                  lcase
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  on-matrix
                  oracle-ridge
                  q2-alpha
                  q2-float32-bound
                  [ridge-solution-coef oracle-coef]
                  rounded
                  weighted-gram
                  window-draw
                  window-draws
                  window-float64-bound)
         (only-in "private/linear-refusal-rows.rkt" window-refusal))

(define n 60)

(define (in-dtype dt vs)
  (map (lambda (v) (inexact->exact (rounded dt v))) vs))

(define (window-case dt narrow wide eps)
  (define base (normal-case n 3 21))
  (define noise (normal-case n 1 77))
  (lcase (format "x1 2^~a, wide 2^~a, eps ~a" narrow wide eps)
         (for/list ([row (in-list (lcase-X base))]
                    [i (in-naturals)]
                    [noise-row (in-list (lcase-X noise))])
           (match-define (list x1 x2 _) row)
           (match-define (list z) noise-row)
           (match-define (list a b) (map (lambda (draw) (window-draw i draw)) window-draws))
           (in-dtype dt
                     (list (* x1 (expt 2 narrow))
                           x2
                           (* a (expt 2 wide))
                           (* b (expt 2 wide))
                           (* (+ a b (* eps z)) (expt 2 wide)))))
         (in-dtype dt (lcase-y base))
         #f))

(define (q2-case dt)
  (define c (normal-case n 4 11))
  (lcase "Q2"
         (for/list ([row (in-list (lcase-X c))])
           (in-dtype dt (map * row (list (expt 2 -13) 1 (expt 2 13) 1))))
         (in-dtype dt (lcase-y c))
         #f))

(define (narrowest c)
  (apply min (map (lambda (col) (for/sum ([x (in-list col)]) (* x x))) (columns-of (lcase-X c)))))

(define (window-alpha c dt k)
  (real->double-flonum (* (expt 2 k) (inexact->exact (unit-roundoff dt)) (narrowest c))))

(define (fit c dt alpha fi)
  (ridge-fit (on-matrix (lcase-X c) dt)
             (on-vector (lcase-y c) dt)
             #:alpha alpha
             #:solver 'svd
             #:fit-intercept fi))

(define (relative-error c dt alpha fi)
  (define c* (oracle-coef (oracle-ridge (lcase-X c) (lcase-y c) #f (inexact->exact alpha) fi)))
  (define ours (map inexact->exact (device-vector->list (ridge-result-coef (fit c dt alpha fi)))))
  (/ (gap ours c*) (max-abs c*)))

(define (regularised-condition c alpha)
  (define-values (G _b _mx _my) (weighted-gram (lcase-X c) (lcase-y c) #f #f))
  (equilibrated-condition (map-diagonal (lambda (g) (+ g (inexact->exact alpha))) G)))

(define float64-refused-share (cons (expt 2 -18) 0.35))

(test-gpu "'svd refuses exactly collinear wide columns for an alpha inside the window"
  (for* ([dt (in-list '(float64 float32))]
         [shape (in-list '((0 13) (0 6) (0 0)))]
         [fi (in-list '(#f #t))]
         [k (in-list '(1 8))])
    (match-define (list narrow wide) shape)
    (define c (window-case dt narrow wide 0))
    (define alpha (window-alpha c dt k))
    (test-case (format "~a ~a at 2^~a u s2min, intercept ~a" dt shape k fi)
      (check-cuml-error 'logic window-refusal (lambda () (fit c dt alpha fi))))))

(test-gpu "'svd keeps fitting Q2 and nearly collinear float64 columns inside the window"
  (for ([fi (in-list '(#f #t))])
    (test-case (format "Q2 float32, intercept ~a" fi)
      (check-true (< (relative-error (q2-case 'float32) 'float32 q2-alpha fi) q2-float32-bound)))
    (for* ([eps (in-list '(1 1/16 1/256))]
           [k (in-list '(1 16))])
      (define c (window-case 'float64 0 13 eps))
      (define alpha (window-alpha c 'float64 k))
      (test-case (format "eps ~a at 2^~a u s2min, intercept ~a" eps k fi)
        (check-true (< (relative-error c 'float64 alpha fi) window-float64-bound))))))

(test-gpu "float64 'svd refuses window fits with u kappa' in [2^-18, 0.35)"
  (match-define (cons share-lo share-hi) float64-refused-share)
  (define u (inexact->exact (unit-roundoff 'float64)))
  (for* ([eps (in-list (list (expt 2 -12) (expt 2 -14) (expt 2 -16) (expt 2 -18)))]
         [k (in-list '(1 16))])
    (define c (window-case 'float64 0 13 eps))
    (define alpha (window-alpha c 'float64 k))
    (match-define (cons kappa-lo kappa-hi) (regularised-condition c alpha))
    (test-case (format "eps ~a at 2^~a u s2min" eps k)
      (check-true (<= share-lo (* u kappa-lo) (* u kappa-hi) share-hi))
      (check-cuml-error 'logic window-refusal (lambda () (fit c 'float64 alpha #f))))))
