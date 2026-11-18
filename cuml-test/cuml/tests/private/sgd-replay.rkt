#lang racket/base

(require (only-in racket/list split-at)
         (only-in racket/math sgn sqr))

(provide (struct-out replay)
         sgd-objective
         sgd-replay)

(struct replay (coef intercept epochs margin) #:transparent)

(define adaptive-floor 1e-6)

(define (dot u v)
  (for/sum ([a (in-list u)] [b (in-list v)]) (* a b)))

(define (axpy a x y)
  (map (lambda (xi yi) (+ (* a xi) yi)) x y))

(define (mean xs)
  (/ (apply + xs) (length xs)))

(define (sigmoid z)
  (/ 1.0 (+ 1.0 (exp (- z)))))

(define (penalty-gradient penalty alpha ratio w)
  (for/list ([v (in-list w)])
    (case penalty
      [(l2) (* 2 alpha v)]
      [(l1) (* alpha (sgn v))]
      [(elasticnet) (+ (* 2 alpha (- 1 ratio) v) (* alpha ratio (sgn v)))]
      [else 0.0])))

(define (penalty-value penalty alpha ratio w)
  (define l1 (for/sum ([v (in-list w)]) (abs v)))
  (define l2 (dot w w))
  (case penalty
    [(l2) (* alpha l2)]
    [(l1) (* alpha l1)]
    [(elasticnet) (* alpha (+ (* (- 1 ratio) l2) (* ratio l1)))]
    [else 0.0]))

(define (residual loss z y)
  (case loss
    [(squared-loss) (* 2 (- z y))]
    [(log) (- (sigmoid z) y)]
    [else
     (if (< (* y z) 1)
         (- y)
         0.0)]))

(define (batch-gradient loss rows ys w)
  (define m (length ys))
  (for/fold ([g (map (lambda (_) 0.0) w)])
            ([x (in-list rows)]
             [y (in-list ys)])
    (axpy (/ (residual loss (dot x w) y) m) x g)))

(define (row-loss loss z y)
  (case loss
    [(squared-loss) (sqr (- y z))]
    [(log)
     (define p (sigmoid z))
     (- (+ (* y (log p)) (* (- 1 y) (log (- 1 p)))))]
    [else (max 0.0 (- 1 (* y z)))]))

(define (sgd-objective loss penalty alpha ratio X y w b)
  (+ (mean (for/list ([x (in-list X)]
                      [v (in-list y)])
             (row-loss loss (+ b (dot x w)) v)))
     (penalty-value penalty alpha ratio w)))

(define (full-loss loss rows ys w)
  (define losses
    (for/list ([x (in-list rows)]
               [y (in-list ys)])
      (row-loss loss (dot x w) y)))
  (if (eq? loss 'hinge)
      (apply + losses)
      (mean losses)))

(define (batches xs size)
  (let loop ([xs xs]
             [acc '()])
    (cond
      [(null? xs) (reverse acc)]
      [(<= (length xs) size) (reverse (cons xs acc))]
      [else
       (define-values (head tail) (split-at xs size))
       (loop tail (cons head acc))])))

(define (sgd-replay X
                    y
                    #:loss [loss 'squared-loss]
                    #:penalty [penalty 'l2]
                    #:alpha [alpha 1e-4]
                    #:l1-ratio [ratio 0.15]
                    #:fit-intercept [fi #t]
                    #:epochs [epochs 1000]
                    #:tol [tol 1e-3]
                    #:learning-rate [schedule 'constant]
                    #:eta0 [eta0 1e-3]
                    #:power-t [power-t 0.5]
                    #:batch-size [size 32]
                    #:n-iter-no-change [patience 5])
  (define rows
    (for/list ([r (in-list X)])
      (map exact->inexact r)))
  (define ys (map exact->inexact y))
  (define mx
    (if fi
        (map mean (apply map list rows))
        (map (lambda (_) 0.0) (car rows))))
  (define my
    (if fi
        (mean ys)
        0.0))
  (define xc
    (for/list ([r (in-list rows)])
      (map - r mx)))
  (define yc (map (lambda (v) (- v my)) ys))
  (define row-batches (batches xc size))
  (define y-batches (batches yc size))
  (define (objective w)
    (+ (full-loss loss xc yc w) (penalty-value penalty alpha ratio w)))
  (define (step w t lr)
    (for/fold ([w w]
               [t t]
               [lr lr])
              ([xb (in-list row-batches)]
               [yb (in-list y-batches)])
      (define rate
        (case schedule
          [(constant) eta0]
          [(invscaling) (/ eta0 (expt (floor t) power-t))]
          [else lr]))
      (define g (map + (batch-gradient loss xb yb w) (penalty-gradient penalty alpha ratio w)))
      (values (axpy (- rate) g w) (add1 t) rate)))
  (define (done w ran margin)
    (replay w (- my (dot mx w)) ran margin))
  (let loop ([e 0]
             [w (map (lambda (_) 0.0) mx)]
             [t 1.0]
             [lr eta0]
             [prev 0.0]
             [count 0]
             [margin +inf.0])
    (cond
      [(= e epochs) (done w e margin)]
      [else
       (define-values (w* t* rate) (step w t lr))
       (define lr* (if (eq? schedule 'adaptive) lr rate))
       (cond
         [(not (> tol 0)) (loop (add1 e) w* t* lr* prev count margin)]
         [else
          (define cur (objective w*))
          (define slack (/ (abs (- cur (- prev tol))) (max (abs cur) 1e-300)))
          (define margin*
            (if (> e 0)
                (min margin slack)
                margin))
          (define worse? (and (> e 0) (> cur (- prev tol))))
          (define count*
            (if worse?
                (add1 count)
                0))
          (cond
            [(<= count* patience) (loop (add1 e) w* t* lr* cur count* margin*)]
            [(and (eq? schedule 'adaptive) (> lr adaptive-floor))
             (loop (add1 e) w* t* (/ lr 5) cur 0 margin*)]
            [else (done w* (add1 e) margin*)])])])))
