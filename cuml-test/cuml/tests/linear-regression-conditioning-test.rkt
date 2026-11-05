#lang racket/base

(require (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/linear-data.rkt"
                  algorithms
                  check-law
                  dyadic
                  exact-fit
                  fit-gap
                  fit-of
                  fit-size
                  gap
                  lcase
                  lcase-X
                  lcase-name
                  lcase-w
                  lcase-y
                  max-abs
                  oracle-lstsq
                  rounded-case
                  solution
                  unit-roundoff))

(define (fit-case c dt a)
  (fit-of (lcase-X c) (lcase-y c) (lcase-w c) dt #:algorithm a))

(define (oracle-of c)
  (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))

(define (offset-case dt offset b1)
  (define draw (uniform-source 3))
  (define X
    (for/list ([_ (in-range 200)])
      (list (+ offset (dyadic (draw))) (dyadic (- (* 2 (draw)) 1)))))
  (rounded-case (lcase (format "offset ~a" offset)
                       X
                       (for/list ([row (in-list X)])
                         (match-define (list x1 x2) row)
                         (+ (* 5 x2) (* b1 (- x1 offset))))
                       #f)
                dt))

(define (offset-x-case offset)
  (define draw (uniform-source 7))
  (define xs
    (for/list ([_ (in-range 200)])
      (+ offset (inexact->exact (draw)))))
  (rounded-case (lcase (format "x offset ~a" offset)
                       (map list xs)
                       (for/list ([x (in-list xs)])
                         (+ 3 (* 2 (- x offset)) (* 1/100 (- (inexact->exact (draw)) 1/2))))
                       #f)
                'float64))

(define (column-means c)
  (define n (length (lcase-y c)))
  (for/list ([col (in-list (apply map list (lcase-X c)))])
    (define mean (/ (apply + col) n))
    (list mean (sqrt (/ (for/sum ([x (in-list col)]) (sqr (- x mean))) n)))))

(test-gpu "a column whose mean dwarfs its spread keeps its direction"
  (for* ([fixture (in-list (list (list 'float64 (offset-case 'float64 #e1e6 #e1e-6))
                                 (list 'float32 (offset-case 'float32 1000 1))
                                 (list 'float64 (offset-x-case #e1e11))))]
         #:do [(match-define (list dt c) fixture)
               (match-define (solution c* b* kappa) (oracle-of c))
               (define u (unit-roundoff dt))
               (define means (column-means c))
               (define ratio (apply max (map (lambda (m) (/ (abs (car m)) (cadr m))) means)))
               (define bound
                 (* (+ (* 128 kappa kappa u) (* 16 (sqr (* u ratio)))) (max 1 (max-abs c*))))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a ~a" dt (lcase-name c) a)
      (match-define (list coef b) (exact-fit (fit-case c dt a)))
      (check-law "T1, plus the centring's second-order term" (gap coef c*) bound)
      (check-law "intercept, the coef bound times the column mean"
                 (abs (- b b*))
                 (* bound (add1 (apply max (map (lambda (m) (abs (car m))) means))))))))

(define low-weight-column
  (let ([t #e1e-12])
    (lcase "low weight"
           '((0 0) (1 0) (2 0) (0 1) (0 1) (1 1))
           '(1 2 3 5 7 7)
           (list 1 1 1 t (* 3 t) (* 2 t)))))

(test-gpu "a column whose variation sits on low-weight rows keeps its direction"
  (for* ([dt (in-list '(float32 float64))]
         #:do [(define c (rounded-case low-weight-column dt))
               (match-define (solution c* b* _) (oracle-of c))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" dt a)
      (define bound
        (* (if (eq? dt 'float32)
               (expt 2 -16)
               (expt 2 -40))
           (fit-size (list c* b*))))
      (check-law "within 2^-16 (float32) or 2^-40 (float64) of the oracle"
                 (fit-gap (fit-case c dt a) (list c* b*))
                 bound))))

(define (offset-y-case offset)
  (define draw (uniform-source 11))
  (define xs
    (for/list ([_ (in-range 200)])
      (inexact->exact (draw))))
  (rounded-case (lcase (format "y offset ~a" offset)
                       (map list xs)
                       (for/list ([x (in-list xs)])
                         (+ offset (* #e1e-6 (+ x (* 1/100 (- (inexact->exact (draw)) 1/2))))))
                       #f)
                'float64))

(define near-constant-y
  (lcase "y = (1 1 1 1+2^-52)" '((0) (1) (2) (3)) (list 1 1 1 (+ 1 (expt 2 -52))) #f))

(test-gpu "a fit is accepted when y's or a column's mean dwarfs its spread"
  (for* ([c (in-list (list (offset-y-case #e1e6) (offset-y-case #e1e7) near-constant-y))]
         #:do [(match-define (solution c* b* kappa) (oracle-of c))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" (lcase-name c) a)
      (define fitted (fit-case c 'float64 a))
      (check-law "T1"
                 (fit-gap fitted (list c* b*))
                 (* 128 (unit-roundoff 'float64) kappa kappa (fit-size (list c* b*)))))))

(define (tiny-weight-cases)
  (define w (inexact->exact 1e-300))
  (list (lcase "one column" '((0) (1) (2) (3)) (list 1 1 1 (+ 1 (expt 2 -52))) (list 1 1 1 w))
        (lcase "two columns"
               '((0 1) (1 0) (2 2) (3 1) (4 0))
               (list 5 5 5 5 (+ 5 (expt 2 -50)))
               (list 1 2 3 1 w))))

(test-gpu "a counted weight whose squares underflow double still fits"
  (for* ([c (in-list (tiny-weight-cases))]
         #:do [(match-define (solution c* b* _) (oracle-of c))]
         [a (in-list algorithms)])
    (test-case (format "~a ~a" (lcase-name c) a)
      (match-define (list coef b) (exact-fit (fit-case c 'float64 a)))
      (check-law "coef within 2^-20 of the exact answer, plus 16 subnormal steps"
                 (gap coef c*)
                 (+ (* (expt 2 -20) (max-abs c*)) (* 16 (expt 2 -1074))))
      (check-law "intercept" (abs (- b b*)) (* 8 (unit-roundoff 'float64) (abs b*))))))
