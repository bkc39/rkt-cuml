#lang racket/base

(require (only-in math/bigfloat
                  bf
                  bf*
                  bf+
                  bf-
                  bf/
                  bf<
                  bigfloat->real
                  bf-precision
                  bfabs
                  bfexp
                  bflog1p
                  bfmax
                  bfsqrt)
         (only-in racket/list make-list)
         (only-in racket/match match))

(provide (struct-out residual)
         residual-of)

(struct residual (K F bound) #:transparent)

(define oracle-bits 128)
(define tol-floor 1/10000)
(define resolved-factor 64)

(define zero (bf 0))
(define one (bf 1))

(define (bfsum xs)
  (apply bf+ zero xs))

(define (exact v)
  (bf (inexact->exact v)))

(define (softplus z)
  (if (bf< zero z)
      (bf+ z (bflog1p (bfexp (bf- zero z))))
      (bflog1p (bfexp z))))

(define (score-of z label)
  (for/first ([s (in-list z)]
              [c (in-naturals)]
              #:when (= c label))
    s))

(define (row-terms z label)
  (match z
    [(list s)
     (define p (bf/ one (bf+ one (bfexp (bf- zero s)))))
     (values (bf- (softplus s) (if (= label 1) s zero))
             (list (if (= label 1)
                       (bf- p one)
                       p)))]
    [(cons first rest)
     (define top
       (for/fold ([m first]) ([s (in-list rest)])
         (bfmax m s)))
     (define shifted (map (lambda (s) (bfexp (bf- s top))) z))
     (define total (bfsum shifted))
     (define lse (bf+ top (bflog1p (bf- total one))))
     (values (bf- lse (score-of z label))
             (for/list ([e (in-list shifted)]
                        [c (in-naturals)])
               (define p (bf/ e total))
               (if (= c label)
                   (bf- p one)
                   p)))]))

(define (penalty-shares penalty ratio)
  (match penalty
    ['l1 (values 1 0)]
    ['l2 (values 0 1)]
    ['elasticnet (values (inexact->exact ratio) (- 1 (inexact->exact ratio)))]
    ['none (values 0 0)]))

(define (column-means X w fi)
  (define total (apply + w))
  (for/list ([column (in-list (apply map list X))])
    (if fi
        (/ (for/sum ([x (in-list column)] [wi (in-list w)])
                    (* (inexact->exact wi) (inexact->exact x)))
           (inexact->exact total))
        0)))

(define (column-scales X w means)
  (define total (apply + w))
  (for/list ([column (in-list (apply map list X))]
             [mean (in-list means)])
    (define square
      (/ (for/sum ([x (in-list column)] [wi (in-list w)]) (* wi (expt (- x mean) 2))) total))
    (and (positive? square) (bfsqrt (bf square)))))

(define (loss-terms rows y weights W b)
  (for/fold ([loss zero]
             [gradient (make-list (length W) (make-list (length (car rows)) zero))]
             [slope (make-list (length W) zero)])
            ([x (in-list rows)]
             [label (in-list y)]
             [wi (in-list weights)])
    (define z
      (for/list ([wc (in-list W)]
                 [bc (in-list b)])
        (bf+ bc (bfsum (map bf* wc x)))))
    (define-values (l dz) (row-terms z label))
    (define weight (bf wi))
    (values (bf+ loss (bf* weight l))
            (for/list ([g (in-list gradient)]
                       [d (in-list dz)])
              (map (lambda (gj xj) (bf+ gj (bf* weight (bf* d xj)))) g x))
            (map (lambda (s d) (bf+ s (bf* weight d))) slope dz))))

(define (penalty-terms penalty ratio C total)
  (define-values (rho rest) (penalty-shares penalty ratio))
  (define scale (bf (/ 1 (* (inexact->exact C) total))))
  (values (bf* (bf rho) scale) (bf* (bf rest) scale)))

(define (projected-gradient g v l1)
  (cond
    [(bf< zero v) (bf+ g l1)]
    [(bf< v zero) (bf- g l1)]
    [else (bfmax zero (bf- (bfabs g) l1))]))

(define (residual-of X
                     y
                     w
                     coef
                     intercept
                     #:penalty [penalty 'l2]
                     #:C [C 1.0]
                     #:l1-ratio [ratio #f]
                     #:fit-intercept [fi #t]
                     #:tol [tol 1e-4])
  (parameterize ([bf-precision oracle-bits])
    (define weights (or w (make-list (length y) 1)))
    (define total (apply + weights))
    (define W (map (lambda (row) (map exact row)) coef))
    (define-values (loss gradient slope)
      (loss-terms (map (lambda (row) (map exact row)) X) y weights W (map exact intercept)))
    (define-values (l1 l2) (penalty-terms penalty ratio C total))
    (define big-total (bf total))
    (define means (column-means X weights fi))
    (define sigma (column-scales X weights means))
    (define-values (F worst)
      (for*/fold ([F (bf/ loss big-total)]
                  [worst (if fi
                             (for/fold ([m zero]) ([s (in-list slope)])
                               (bfmax m (bfabs (bf/ s big-total))))
                             zero)])
                 ([(wc gc sc) (in-parallel (in-list W) (in-list gradient) (in-list slope))]
                  [(v g0 s mean)
                   (in-parallel (in-list wc) (in-list gc) (in-list sigma) (in-list means))])
        (define g (bf+ (bf/ (bf- g0 (bf* (bf mean) sc)) big-total) (bf* l2 v)))
        (values (bf+ F (bf+ (bf* l1 (bfabs v)) (bf* (bf/ l2 (bf 2)) (bf* v v))))
                (if s
                    (bfmax worst (bf/ (bfabs (projected-gradient g v l1)) s))
                    worst))))
    (define tau (max (inexact->exact tol) tol-floor))
    (define F* (bigfloat->real F))
    (residual (bigfloat->real worst) F* (* resolved-factor tau (max F* tau)))))
