#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-equal? check-pred test-case)
         (only-in "private/linear-centring-data.rkt"
                  case-staging
                  consistent
                  exact-norm
                  failed-laws
                  fails?
                  staged-bottom-vector)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y oracle-lstsq solution-coef)
         (only-in "private/linear-laws.rkt" staging)
         (only-in "private/real-linear-data.rkt" offset-law-arms real-lcase))

(for* ([dt (in-list '(float32 float64))]
       [arm (in-list (hash-ref offset-law-arms dt))])
  (test-case (format "breast-cancer ~a ~a: c* passes; 0.9 c* fails a law, a halved intercept T4~a"
                     arm
                     dt
                     (if (eq? dt 'float64)
                         ", c* off by 1/10 of its norm along G~'s bottom eigenvector T1"
                         ""))
    (define c (real-lcase 'breast-cancer arm dt))
    (define exact (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) #t))
    (define c* (solution-coef exact))
    (match-define (list rounded b) (consistent c c*))
    (check-equal? (failed-laws c dt (list rounded b) exact) '())
    (check-pred pair? (failed-laws c dt (consistent c (map (lambda (cj) (* 9/10 cj)) c*)) exact))
    (check-pred (fails? "T4") (failed-laws c dt (list rounded (/ b 2)) exact))
    (when (eq? dt 'float64)
      (match-define (staging scale y-scale _ _ _ _ _) (case-staging c))
      (define v (staged-bottom-vector c))
      (define step
        (/ (exact-norm (map (lambda (cj dj) (/ (* y-scale cj) dj)) c* scale)) (* 10 (exact-norm v))))
      (define moved
        (for/list ([cj (in-list c*)]
                   [vj (in-list v)]
                   [dj (in-list scale)])
          (+ cj (/ (* step vj dj) y-scale))))
      (check-pred (fails? "T1") (failed-laws c dt (consistent c moved) exact)))))
