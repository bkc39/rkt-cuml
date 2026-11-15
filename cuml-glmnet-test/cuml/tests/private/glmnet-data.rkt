#lang racket/base

(require (only-in cuml/tests/private/linear-data
                  by-dtype
                  lcase
                  lcase-name
                  rounded
                  scaled-case
                  weighted-spread
                  weights-or-ones)
         (only-in cuml/tests/private/logistic-data lr-case lr-case-X lr-fit setting)
         (only-in glmnet
                  elnet-fit
                  elnet-result-coefficients
                  elnet-result-intercept
                  logistic-fit
                  logistic-result-coefficients
                  logistic-result-intercept
                  multinomial-fit
                  multinomial-predict-proba
                  multinomial-result-coefficients
                  multinomial-result-intercepts)
         (only-in racket/list append* make-list)
         (only-in racket/match match match-define struct*)
         (only-in threading ~>))

(provide bound-of
         default-bound
         glmnet-binomial
         glmnet-gaussian
         glmnet-multinomial
         narrowed-lr-case
         trial
         trial-case
         trial-Cs
         whole-weighted)

(struct trial
        (case Cs
          float32
          float64
          default)
  #:name trial-row
  #:constructor-name trial-row)

(define (trial c #:Cs [Cs '()] #:float32 float32 #:float64 float64 #:default [default #f])
  (trial-row c Cs float32 float64 default))

(define (bound-of t dt)
  (by-dtype dt (trial-float32 t) (trial-float64 t)))

(define (default-bound t dt)
  (define cap (trial-default t))
  (and cap (cap dt)))

(define logistic-thresh 1e-20)
;; glmnet's multinomial l1 on iris at C 1 with an intercept never converges below 1e-18
(define multinomial-thresh 1e-18)
(define gaussian-thresh 1e-26)
(define glmnet-max-iters 10000000)
(define weight-scale 64)

(define (flonum-rows rows)
  (for/list ([row (in-list rows)])
    (map exact->inexact row)))

(define (narrowed-lr-case c dt)
  (define (narrowed row)
    (map (lambda (x) (inexact->exact (rounded dt x))) row))
  (struct-copy lr-case c [X (map narrowed (lr-case-X c))]))

(define (whole-weighted c)
  (struct-copy lcase
               (scaled-case c #:w weight-scale)
               [name (format "~a, weights times ~a" (lcase-name c) weight-scale)]))

(define (replicated X y w)
  (define counts (weights-or-ones w y))
  (define (each xs)
    (append* (map make-list counts xs)))
  (values (each X) (each y) (apply + counts)))

(define (binomial-lambda C total)
  (exact->inexact (/ 1 (* (inexact->exact C) total))))

(define (setting-alpha s)
  (match s
    [(setting 'l2 _ _) 0.0]
    [(setting 'l1 _ _) 1.0]
    [(setting 'elasticnet _ ratio) (exact->inexact ratio)]))

(define (glmnet-binomial c s #:fit-intercept [fi #t])
  (match-define (struct* lr-case ([X X] [y y] [w w])) c)
  (match-define (struct* setting ([C C])) s)
  (define-values (rows labels total) (replicated X y w))
  (define r
    (logistic-fit (flonum-rows rows)
                  labels
                  #:lambda (binomial-lambda C total)
                  #:alpha (setting-alpha s)
                  #:standardize? #f
                  #:intercept? fi
                  #:thresh logistic-thresh
                  #:max-iters glmnet-max-iters))
  (lr-fit (~> r logistic-result-coefficients vector->list list)
          (list (logistic-result-intercept r))
          #f))

(define (glmnet-multinomial c s #:fit-intercept [fi #t])
  (match-define (struct* lr-case ([X X] [y y] [w w])) c)
  (match-define (struct* setting ([C C])) s)
  (define-values (rows labels total) (replicated X y w))
  (define r
    (multinomial-fit (flonum-rows rows)
                     labels
                     #:lambda (binomial-lambda C total)
                     #:alpha (setting-alpha s)
                     #:standardize? #f
                     #:intercept? fi
                     #:thresh multinomial-thresh
                     #:max-iters glmnet-max-iters))
  (values (lr-fit (~> r multinomial-result-coefficients vector->list (map vector->list _))
                  (vector->list (multinomial-result-intercepts r))
                  #f)
          (multinomial-predict-proba r (flonum-rows X))))

(define (y-spread y w fi)
  (define weights (weights-or-ones w y))
  (define squares
    (if fi
        (weighted-spread y weights)
        (for/sum ([v (in-list y)] [wi (in-list weights)]) (* wi v v))))
  (~> (/ squares (apply + weights)) exact->inexact sqrt))

(define (glmnet-gaussian c alpha l1-ratio #:fit-intercept [fi #t])
  (match-define (struct* lcase ([X X] [y y] [w w])) c)
  (define-values (rows ys _) (replicated X y w))
  (define l1 (* alpha l1-ratio))
  (define lam (+ l1 (* (y-spread y w fi) alpha (- 1 l1-ratio))))
  (define r
    (elnet-fit (flonum-rows rows)
               (map exact->inexact ys)
               #:lambda (exact->inexact lam)
               #:alpha (exact->inexact (/ l1 lam))
               #:standardize? #f
               #:intercept? fi
               #:thresh gaussian-thresh
               #:max-iters glmnet-max-iters))
  (lr-fit (~> r elnet-result-coefficients vector->list list) (list (elnet-result-intercept r)) #f))
