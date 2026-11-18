#lang racket/base

(require (only-in cuml/linear-model ridge-fit ridge-result-coef)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt"
                  columns-of
                  gap
                  lcase
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  on-matrix
                  on-vector
                  oracle-ridge
                  rounded
                  unit-roundoff)
         (only-in "private/linear-data.rkt" [ridge-solution-coef oracle-coef]))

(define n 60)

(define (in-dtype dt v)
  (inexact->exact (rounded dt v)))

(define (window-case dt narrow wide eps)
  (define base (normal-case n 3 21))
  (define noise (normal-case n 1 77))
  (lcase (format "x1 2^~a, wide 2^~a, eps ~a" narrow wide eps)
         (for/list ([row (in-list (lcase-X base))]
                    [i (in-naturals)]
                    [z (in-list (lcase-X noise))])
           (match-define (list x1 x2 _) row)
           (define a (- (modulo (* 37 (+ i 3)) 101) 50))
           (define b (- (modulo (* 53 (+ i 7)) 97) 48))
           (for/list ([v (in-list (list (* x1 (expt 2 narrow))
                                        x2
                                        (* a (expt 2 wide))
                                        (* b (expt 2 wide))
                                        (* (+ a b (* eps (car z))) (expt 2 wide))))])
             (in-dtype dt v)))
         (for/list ([v (in-list (lcase-y base))])
           (in-dtype dt v))
         #f))

(define (q2-case dt)
  (define c (normal-case 60 4 11))
  (lcase "Q2"
         (for/list ([row (in-list (lcase-X c))])
           (for/list ([x (in-list row)]
                      [f (in-list (list (expt 2 -13) 1 (expt 2 13) 1))])
             (in-dtype dt (* x f))))
         (for/list ([v (in-list (lcase-y c))])
           (in-dtype dt v))
         #f))

(define (narrowest c)
  (for/fold ([lo #f]) ([col (in-list (columns-of (lcase-X c)))])
    (define s (for/sum ([x (in-list col)]) (* x x)))
    (if lo
        (min lo s)
        s)))

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

(define window-refusal
  #rx"^ridge-fit: X: the svd solve's regularised Gram matrix has u[*]kappa (inf|[0-9.e+-]+) >= 2\\^-[0-9]+; X is ill-conditioned for this alpha [(].*a larger alpha may fit[)]$")

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
      (check-true (< (relative-error (q2-case 'float32) 'float32 1.0 fi) (expt 2 -10))))
    (for* ([eps (in-list '(1 1/16 1/256))]
           [k (in-list '(1 16))])
      (define c (window-case 'float64 0 13 eps))
      (define alpha (window-alpha c 'float64 k))
      (test-case (format "eps ~a at 2^~a u s2min, intercept ~a" eps k fi)
        (check-true (< (relative-error c 'float64 alpha fi) 1e-10))))))
