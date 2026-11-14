#lang racket/base

(require (only-in math/bigfloat
                  0.bf
                  1.bf
                  bf
                  bf*
                  bf+
                  bf-
                  bf/
                  bf<
                  bf>
                  bigfloat->real
                  bf-precision
                  bfabs
                  bfexp
                  bflog1p
                  bfmax
                  bfmin
                  bfsqrt)
         (only-in racket/list make-list)
         (only-in racket/match match)
         (only-in "linear-data.rkt" columns-of))

(provide (struct-out residual)
         residual-of)

(struct residual (gradient objective bound) #:transparent)

(define oracle-bits 128)
(define tolerance-floor 1/10000)
(define residual-factor 64)
(define constant-spread 16)

(define (bfsum xs)
  (apply bf+ 0.bf xs))

(define (exact v)
  (bf (inexact->exact v)))

(define (exact-rows rows)
  (map (lambda (row) (map exact row)) rows))

(define (softplus z)
  (if (bf< 0.bf z)
      (bf+ z (bflog1p (bfexp (bf- 0.bf z))))
      (bflog1p (bfexp z))))

(define (row-terms z label)
  (match z
    [(list s)
     (define p (bf/ 1.bf (bf+ 1.bf (bfexp (bf- 0.bf s)))))
     (values (bf- (softplus s) (if (= label 1) s 0.bf))
             (list (if (= label 1)
                       (bf- p 1.bf)
                       p)))]
    [(cons first rest)
     (define top
       (for/fold ([m first]) ([s (in-list rest)])
         (bfmax m s)))
     (define shifted (map (lambda (s) (bfexp (bf- s top))) z))
     (define total (bfsum shifted))
     (define lse (bf+ top (bflog1p (bf- total 1.bf))))
     (values (bf- lse (list-ref z label))
             (for/list ([e (in-list shifted)]
                        [c (in-naturals)])
               (define p (bf/ e total))
               (if (= c label)
                   (bf- p 1.bf)
                   p)))]))

(define (penalty-shares penalty ratio)
  (match penalty
    ['l1 (values 1 0)]
    ['l2 (values 0 1)]
    ['elasticnet (values (inexact->exact ratio) (- 1 (inexact->exact ratio)))]
    ['none (values 0 0)]))

(define (column-centres X w)
  (define total (apply + w))
  (for/list ([column (in-list (columns-of X))])
    (/ (for/sum ([x (in-list column)] [wi (in-list w)])
                (* (inexact->exact wi) (inexact->exact x)))
       (inexact->exact total))))

(define (column-magnitudes X w)
  (for/list ([column (in-list (columns-of X))])
    (for/fold ([most 0])
              ([x (in-list column)]
               [wi (in-list w)]
               #:when (positive? wi))
      (max most (abs (inexact->exact x))))))

(define (column-scales X w means)
  (define total (apply + w))
  (for/list ([column (in-list (columns-of X))]
             [mean (in-list means)])
    (define square
      (/ (for/sum ([x (in-list column)] [wi (in-list w)]) (* wi (expt (- x mean) 2))) total))
    (and (positive? square) (bfsqrt (bf square)))))

(define (unit-roundoff dt)
  (match dt
    ['float32 (expt 2 -24)]
    ['float64 (expt 2 -53)]))

(define (residual-spreads X w centres fi dt)
  (define centred (column-scales X w centres))
  (if fi
      centred
      (for/list ([c (in-list centred)]
                 [rms (in-list (column-scales X w (make-list (length centres) 0)))]
                 [most (in-list (column-magnitudes X w))])
        (if (and c (bf> c (bf (* constant-spread (unit-roundoff dt) most))))
            (bfmin c rms)
            rms))))

(define (loss-terms rows y weights W b)
  (for/fold ([loss 0.bf]
             [gradient (make-list (length W) (make-list (length (car rows)) 0.bf))]
             [slope (make-list (length W) 0.bf)])
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
    [(bf< 0.bf v) (bf+ g l1)]
    [(bf< v 0.bf) (bf- g l1)]
    [else (bfmax 0.bf (bf- (bfabs g) l1))]))

(define (intercept-worst slope total fi)
  (if fi
      (for/fold ([m 0.bf]) ([s (in-list slope)])
        (bfmax m (bfabs (bf/ s total))))
      0.bf))

(define (coefficient-terms v g0 sc sigma mean l1 l2 total)
  (define g (bf+ (bf/ (bf- g0 (bf* (bf mean) sc)) total) (bf* l2 v)))
  (values (bf+ (bf* l1 (bfabs v)) (bf* (bf/ l2 (bf 2)) (bf* v v)))
          (and sigma (bf/ (bfabs (projected-gradient g v l1)) sigma))))

(define (residual-of X
                     y
                     w
                     coef
                     intercept
                     #:penalty [penalty 'l2]
                     #:C [C 1.0]
                     #:l1-ratio [ratio #f]
                     #:fit-intercept [fi #t]
                     #:tol [tol 1e-4]
                     #:dtype [dt 'float64])
  (parameterize ([bf-precision oracle-bits])
    (define weights (or w (make-list (length y) 1)))
    (define total (apply + weights))
    (define W (exact-rows coef))
    (define-values (loss gradient slope)
      (loss-terms (exact-rows X) y weights W (map exact intercept)))
    (define-values (l1 l2) (penalty-terms penalty ratio C total))
    (define big-total (bf total))
    (define centres (column-centres X weights))
    (define means
      (if fi
          centres
          (make-list (length centres) 0)))
    (define sigma (residual-spreads X weights centres fi dt))
    (define-values (F worst)
      (for*/fold ([F (bf/ loss big-total)]
                  [worst (intercept-worst slope big-total fi)])
                 ([(wc gc sc) (in-parallel (in-list W) (in-list gradient) (in-list slope))]
                  [(v g0 s mean)
                   (in-parallel (in-list wc) (in-list gc) (in-list sigma) (in-list means))])
        (define-values (term scaled) (coefficient-terms v g0 sc s mean l1 l2 big-total))
        (values (bf+ F term)
                (if scaled
                    (bfmax worst scaled)
                    worst))))
    (define tau (max (inexact->exact tol) tolerance-floor))
    (define F* (bigfloat->real F))
    (residual (bigfloat->real worst) F* (* residual-factor tau (max F* tau)))))
