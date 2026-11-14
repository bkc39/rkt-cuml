#lang racket/base

(require (only-in glmnet
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
         (only-in racket/flonum flsingle)
         (only-in racket/list append* make-list)
         (only-in racket/match match match-define struct*)
         (only-in racket/math sqr)
         (only-in "linear-data.rkt" lcase lcase-name lcase-w lcase-X lcase-y)
         (only-in "logistic-data.rkt" lr-case lr-case-w lr-case-X lr-case-y lr-fit setting))

(provide binomial-lambda
         elnet-mapping
         glmnet-binomial
         glmnet-gaussian
         glmnet-multinomial
         integer-weighted
         narrowed-lcase
         narrowed-lr-case
         replicated
         y-spread)

(define glmnet-thresh 1e-14)
(define glmnet-max-iters 10000000)
(define weight-levels 4)

(define (flonum-rows rows)
  (for/list ([row (in-list rows)])
    (map exact->inexact row)))

(define (narrowed x dt)
  (inexact->exact (if (eq? dt 'float32)
                      (flsingle (exact->inexact x))
                      (exact->inexact x))))

(define (narrowed-rows rows dt)
  (for/list ([row (in-list rows)])
    (map (lambda (x) (narrowed x dt)) row)))

(define (narrowed-lr-case c dt)
  (struct-copy lr-case c [X (narrowed-rows (lr-case-X c) dt)]))

(define (narrowed-lcase c dt)
  (struct-copy lcase
               c
               [X (narrowed-rows (lcase-X c) dt)]
               [y (map (lambda (v) (narrowed v dt)) (lcase-y c))]))

(define (replicated X y w)
  (define counts (or w (make-list (length y) 1)))
  (values (append* (for/list ([row (in-list X)]
                              [k (in-list counts)])
                     (make-list k row)))
          (append* (for/list ([v (in-list y)]
                              [k (in-list counts)])
                     (make-list k v)))
          (apply + counts)))

(define (integer-weighted c seed)
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed seed)
    (struct-copy lcase
                 c
                 [name (format "~a, integer weights" (lcase-name c))]
                 [w
                  (for/list ([_ (in-list (lcase-y c))])
                    (+ 1 (random weight-levels)))])))

(define (binomial-lambda C total)
  (exact->inexact (/ 1 (* (inexact->exact C) total))))

(define (setting-alpha s)
  (match s
    [(setting 'l2 _ _) 0.0]
    [(setting 'l1 _ _) 1.0]
    [(setting 'elasticnet _ ratio) (exact->inexact ratio)]))

(define (glmnet-binomial c s #:fit-intercept [fi #t] #:lambda-scale [scale 1])
  (match-define (struct* lr-case ([X X] [y y] [w w])) c)
  (match-define (struct* setting ([C C])) s)
  (define-values (rows labels total) (replicated X y w))
  (define r
    (logistic-fit (flonum-rows rows)
                  labels
                  #:lambda (* scale (binomial-lambda C total))
                  #:alpha (setting-alpha s)
                  #:standardize? #f
                  #:intercept? fi
                  #:thresh glmnet-thresh
                  #:max-iters glmnet-max-iters))
  (lr-fit (list (vector->list (logistic-result-coefficients r)))
          (list (logistic-result-intercept r))
          #f))

(define (glmnet-multinomial c s #:fit-intercept [fi #t])
  (match-define (struct* lr-case ([X X] [y y])) c)
  (match-define (struct* setting ([C C])) s)
  (define r
    (multinomial-fit (flonum-rows X)
                     y
                     #:lambda (binomial-lambda C (length y))
                     #:alpha (setting-alpha s)
                     #:standardize? #f
                     #:intercept? fi
                     #:thresh glmnet-thresh
                     #:max-iters glmnet-max-iters))
  (values (lr-fit (map vector->list (vector->list (multinomial-result-coefficients r)))
                  (vector->list (multinomial-result-intercepts r))
                  #f)
          (multinomial-predict-proba r (flonum-rows X))))

(define (y-spread y w fi)
  (define weights (or w (make-list (length y) 1)))
  (define total (apply + weights))
  (define mean
    (if fi
        (/ (for/sum ([v (in-list y)] [wi (in-list weights)]) (* wi v)) total)
        0))
  (sqrt (exact->inexact (/ (for/sum ([v (in-list y)] [wi (in-list weights)]) (* wi (sqr (- v mean))))
                           total))))

(define (elnet-mapping alpha ratio spread)
  (define l1 (* alpha ratio))
  (define lam (+ l1 (* spread alpha (- 1 ratio))))
  (values (exact->inexact lam) (exact->inexact (/ l1 lam))))

(define (glmnet-gaussian c alpha ratio #:fit-intercept [fi #t] #:spread [spread #f])
  (match-define (struct* lcase ([X X] [y y] [w w])) c)
  (define-values (rows ys _) (replicated X y w))
  (define-values (lam a) (elnet-mapping alpha ratio (or spread (y-spread y w fi))))
  (define r
    (elnet-fit (flonum-rows rows)
               (map exact->inexact ys)
               #:lambda lam
               #:alpha a
               #:standardize? #f
               #:intercept? fi
               #:thresh glmnet-thresh
               #:max-iters glmnet-max-iters))
  (append (vector->list (elnet-result-coefficients r)) (list (elnet-result-intercept r))))
