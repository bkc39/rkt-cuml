#lang racket/base

(require (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt"
                  check-accuracy
                  coef-distance
                  correlated-case
                  en-case-fit
                  en-run
                  law-alpha
                  norm2)
         (only-in "private/elastic-net-oracle.rkt"
                  curvatures
                  en-problem-total
                  en-solution
                  host-kkt
                  kkt-residual
                  oracle-elastic-net
                  problem-of
                  s2-rounding)
         (only-in "private/linear-data.rkt"
                  check-law
                  dtypes
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  oracle-ridge
                  ridge-solution
                  scaled-case
                  unit-roundoff))

(test-gpu "small l1-ratio approaches ridge at alpha times the weight sum"
  (define c (correlated-case 200 7 0.9 51 #:weighted? #t))
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define alpha (law-alpha p 0.01 1))
  (match-define (struct* ridge-solution ([coef ridge-coef]))
    (oracle-ridge (lcase-X c)
                  (lcase-y c)
                  (lcase-w c)
                  (* (inexact->exact alpha) (en-problem-total p))
                  #t))
  (for* ([dt (in-list dtypes)]
         [r (in-list '(0 1e-6 1e-3 0.1))])
    (test-case (format "~a l1-ratio ~a" dt r)
      (match-define (struct* en-solution ([mu mu] [sigma sigma])) (oracle-elastic-net p alpha r))
      (match-define (struct* en-run ([coef coef] [n-iter k]))
        (en-case-fit c dt #:alpha alpha #:l1-ratio r #:tol 1e-8 #:max-iter 100000))
      (define K (kkt-residual (host-kkt p coef alpha r 1e-8 k 200 (unit-roundoff dt))))
      (check-law "ridge-distance"
                 (coef-distance coef ridge-coef)
                 (+ (/ (* alpha r (sqrt 7)) mu)
                    (/ (* K (max-abs coef) (norm2 (curvatures sigma (* alpha (- 1 r))))) mu)
                    (* s2-rounding (unit-roundoff dt) (norm2 (map exact->inexact ridge-coef))))))))

(test-gpu "one column and wide 8x20 fits meet their accuracy bound"
  (define one (correlated-case 200 1 0 52))
  (define wide (correlated-case 8 20 0 53))
  (for ([dt (in-list dtypes)])
    (define p1 (problem-of (lcase-X one) (lcase-y one) #f #t))
    (check-equal? (check-accuracy one dt (law-alpha p1 0.01 1) 1 1e-3) "coefficient-distance")
    (define pw (problem-of (lcase-X wide) (lcase-y wide) #f #t))
    (for ([r (in-list '(1 0.5))]
          [law (in-list '("kkt-residual" "coefficient-distance"))])
      (check-equal? (check-accuracy wide dt (law-alpha pw 0.1 r) r 1e-8 #:max-iter 100000) law))))

(test-gpu "columns at 1e-4 and 1e-8 meet coefficient-distance at 100 rows"
  (define c (correlated-case 100 4 0.3 54))
  (for* ([dt (in-list dtypes)]
         [f (in-list '(1/10000 1/100000000))]
         [r (in-list '(1 0.5))])
    (define small (scaled-case c #:X f))
    (define p (problem-of (lcase-X small) (lcase-y small) #f #t))
    (check-equal? (check-accuracy small dt (law-alpha p 0.01 r) r 1e-3) "coefficient-distance")))

(test-gpu "a heavy L2 penalty's one-epoch stop is accepted"
  (define c (correlated-case 200 7 0.9 31))
  (for* ([dt (in-list dtypes)]
         [alpha (in-list '(1e4 1e6))])
    (test-case (format "~a alpha ~a" dt alpha)
      (match-define (struct* en-run ([coef coef] [n-iter k]))
        (en-case-fit c dt #:alpha alpha #:l1-ratio 0))
      (check-equal? k 1)
      (check-true (< (max-abs coef) 1e-3))
      (check-equal? (check-accuracy c dt alpha 0 1e-3) "coefficient-distance"))))
