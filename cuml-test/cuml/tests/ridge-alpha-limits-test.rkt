#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt"
                  case-fit
                  check-law
                  dtypes
                  exact-fit
                  gap
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  normal-case
                  ridge-oracles
                  ridge-solution
                  unit-roundoff
                  weights-or-ones)
         (only-in "private/ridge-data.rkt" lstsq-algorithm norm2 ridge-case-fit solvers))

(define cases
  (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t) (normal-case 20 2 1)))

(define (weighted-mean xs ws)
  (/ (for/sum ([x (in-list xs)] [w (in-list ws)]) (* x w)) (apply + ws)))

(test-gpu "as alpha goes to 0 the fit goes to linear-regression-fit's (R5)"
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))]
         [o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi
                                    (list (expt 2 -40) (expt 2 -20) (expt 2 -10))))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (match-define (ridge-solution alpha _ _ _ top bottom _) o)
    (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
      (match-define (list coef _)
        (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
      (match-define (list lstsq _)
        (exact-fit (case-fit c dt #:algorithm (lstsq-algorithm s) #:fit-intercept fi)))
      (check-law "R5"
                 (norm2 (map - coef lstsq))
                 (* (+ (/ alpha (+ bottom alpha)) (* 256 (unit-roundoff dt) (/ top bottom)))
                    (norm2 lstsq))))))

(test-gpu "as alpha grows the coefficients shrink to 0 and the intercept to y's mean (R6)"
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))]
         [o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi
                                    (list 1 (expt 2 20) (expt 2 60))))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (match-define (ridge-solution alpha _ _ kappa top bottom xty) o)
    (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
      (define u (unit-roundoff dt))
      (match-define (list coef b)
        (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
      (check-law "R6" (norm2 coef) (* (/ xty (+ bottom alpha)) (+ 1 (* 64 u kappa))))
      (when (and fi (> alpha (* (expt 2 59) top)))
        (check-law "R6 intercept"
                   (abs (- b (weighted-mean (lcase-y c) (weights-or-ones (lcase-w c) (lcase-y c)))))
                   (* 8 u (max-abs (lcase-y c))))))))

(define kept-one
  (let ([c (normal-case 200 1 3 #:weighted? #t)])
    (lcase "one of three columns kept"
           (for/list ([row (in-list (lcase-X c))])
             (list 5/4 (car row) -3))
           (lcase-y c)
           (lcase-w c))))

(define routed
  (list (list (normal-case 200 1 2) solvers)
        (list kept-one solvers)
        (list (normal-case 5 8 4) '(svd))
        (list (normal-case 6 8 5 #:weighted? #t) '(svd))))

(test-gpu "one kept column and svd on wide X are routed and still solve the ridge problem (R7)"
  (for* ([r (in-list routed)]
         #:do [(match-define (list c ss) r)]
         [fi (in-list '(#t))]
         [o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi (list (expt 2 -7) 1)))]
         [dt (in-list dtypes)]
         [s (in-list ss)])
    (match-define (ridge-solution alpha c* b* kappa _ _ _) o)
    (test-case (format "~a alpha ~a ~a ~a" (lcase-name c) alpha dt s)
      (define u (unit-roundoff dt))
      (match-define (list coef b) (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s)))
      (check-law "R1" (gap coef c*) (* 128 u kappa (max-abs c*)))
      (check-law "R2" (abs (- b b*)) (* 8 u kappa (max-abs (lcase-y c)))))))
