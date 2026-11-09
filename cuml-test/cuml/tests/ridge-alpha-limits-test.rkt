#lang racket/base

(require (only-in racket/match match-define struct*)
         (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
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
                  oracle-ridge
                  ridge-oracles
                  ridge-solution
                  unit-roundoff
                  weights-or-ones)
         (only-in "private/ridge-data.rkt"
                  lstsq-algorithm
                  norm2
                  ridge-case-fit
                  rounding-alpha
                  solvers
                  widest-spread))

(define cases (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t) (normal-case 20 2 1)))

(define (weighted-mean xs ws)
  (/ (for/sum ([x (in-list xs)] [w (in-list ws)]) (* x w)) (apply + ws)))

(define (check-ridge-solution c dt s fi alpha)
  (match-define (struct* ridge-solution ([coef c*] [intercept b*] [kappa kappa]))
    (oracle-ridge (lcase-X c) (lcase-y c) (lcase-w c) alpha fi))
  (define u (unit-roundoff dt))
  (match-define (list coef b)
    (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
  (check-law "R1" (gap coef c*) (* 128 u kappa (max-abs c*)))
  (when fi
    (check-law "R2" (abs (- b b*)) (* 8 u kappa (max-abs (lcase-y c))))))

(test-gpu "as alpha goes to 0 the fit goes to linear-regression-fit's (R5)"
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))]
         [o (in-list (ridge-oracles (lcase-X c)
                                    (lcase-y c)
                                    (lcase-w c)
                                    fi
                                    (list (expt 2 -40) (expt 2 -20) (expt 2 -10))))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (match-define (struct* ridge-solution ([alpha alpha] [top top] [bottom bottom])) o)
    (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
      (match-define (list coef _)
        (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
      (match-define (list lstsq _)
        (exact-fit (case-fit c dt #:algorithm (lstsq-algorithm s) #:fit-intercept fi)))
      (check-law "R5"
                 (norm2 (map - coef lstsq))
                 (* (+ (/ alpha (+ bottom alpha)) (* 256 (unit-roundoff dt) (/ top bottom)))
                    (norm2 lstsq))))))

(test-gpu "an alpha either side of rounding solves the ridge problem"
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))]
         [dt (in-list dtypes)]
         [s (in-list solvers)]
         [factor (in-list '(1/2 2))])
    (test-case (format "~a ~a ~a ~a ~a" (lcase-name c) fi dt s factor)
      (check-ridge-solution c dt s fi (* factor (rounding-alpha c dt fi))))))

(test-gpu "as alpha grows the coefficients shrink to 0 and the intercept to y's mean (R6)"
  (for* ([c (in-list cases)]
         [fi (in-list '(#t #f))]
         [o (in-list
             (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) fi (list 1 (expt 2 20) (expt 2 60))))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (match-define (struct* ridge-solution
                           ([alpha alpha] [kappa kappa] [top top] [bottom bottom] [xty xty]))
      o)
    (test-case (format "~a ~a alpha ~a ~a ~a" (lcase-name c) fi alpha dt s)
      (define u (unit-roundoff dt))
      (match-define (list coef b)
        (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))
      (check-law "R6" (norm2 coef) (* (/ xty (+ bottom alpha)) (+ 1 (* 64 u kappa))))
      (when (and fi (> alpha (* (expt 2 59) top)))
        (check-law "R6 intercept"
                   (abs (- b (weighted-mean (lcase-y c) (weights-or-ones (lcase-w c) (lcase-y c)))))
                   (* 8 u (max-abs (lcase-y c))))))))

(define (huge-ratios dt)
  (if (eq? dt 'float32)
      '(100 109)
      '(210 957)))

(define (refused-ratio dt)
  (if (eq? dt 'float32) 113 961))

(test-gpu "an alpha far past the widest spread still solves the ridge problem (R6)"
  (for* ([c (in-list (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t)))]
         [fi (in-list '(#t #f))]
         [dt (in-list dtypes)]
         [s (in-list solvers)]
         [k (in-list (huge-ratios dt))])
    (test-case (format "~a ~a ~a ~a 2^~a" (lcase-name c) fi dt s k)
      (check-ridge-solution c dt s fi (* (expt 2 k) (widest-spread c fi))))))

(define (check-alpha-refused c dt s fi alpha)
  (test-case (format "~a ~a ~a ~a" (lcase-name c) dt s fi)
    (check-cuml-error
     'logic
     #rx"^ridge-fit: alpha: expected at most [0-9.e+]+ for this X in float(32|64), got "
     (lambda () (ridge-case-fit c dt #:alpha alpha #:solver s #:fit-intercept fi)))))

(test-gpu "an alpha whose staged value overflows the dtype is refused"
  (for* ([c (in-list (list (normal-case 200 7 0) (normal-case 200 7 1 #:weighted? #t)))]
         [fi (in-list '(#t #f))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (check-alpha-refused c dt s fi (* (expt 2 (refused-ratio dt)) (widest-spread c fi)))))

(test-gpu "an alpha whose staged value overflows is refused for a constant y too"
  (define c (normal-case 200 7 0))
  (define constant (lcase "constant y" (lcase-X c) (map (lambda (_) 5/2) (lcase-y c)) #f))
  (for* ([fi (in-list '(#t #f))]
         [dt (in-list dtypes)]
         [s (in-list solvers)])
    (check-alpha-refused constant dt s fi (* (expt 2 (refused-ratio dt)) (widest-spread c fi)))))

(define kept-one
  (let ([c (normal-case 200 1 3 #:weighted? #t)])
    (lcase "one of three columns kept"
           (for/list ([row (in-list (lcase-X c))])
             (list 5/4 (car row) -3))
           (lcase-y c)
           (lcase-w c))))

(struct routing (case solvers))

(define routed
  (list (routing (normal-case 200 1 2) solvers)
        (routing kept-one solvers)
        (routing (normal-case 5 8 4) '(svd))
        (routing (normal-case 6 8 5 #:weighted? #t) '(svd))))

(test-gpu "routed shapes still solve the ridge problem (R7)"
  (for* ([r (in-list routed)]
         #:do [(match-define (routing c ss) r)]
         [o (in-list (ridge-oracles (lcase-X c) (lcase-y c) (lcase-w c) #t (list (expt 2 -7) 1)))]
         [dt (in-list dtypes)]
         [s (in-list ss)])
    (match-define (struct* ridge-solution ([alpha alpha] [coef c*] [intercept b*] [kappa kappa])) o)
    (test-case (format "~a alpha ~a ~a ~a" (lcase-name c) alpha dt s)
      (define u (unit-roundoff dt))
      (match-define (list coef b) (exact-fit (ridge-case-fit c dt #:alpha alpha #:solver s)))
      (check-law "R1" (gap coef c*) (* 128 u kappa (max-abs c*)))
      (check-law "R2" (abs (- b b*)) (* 8 u kappa (max-abs (lcase-y c)))))))
