#lang racket/base

(require (only-in racket/flonum
                  fl*
                  fl+
                  fl-
                  fl/
                  fl<=
                  fl>=
                  flabs
                  flexp
                  flexpt
                  fllog
                  for/flvector
                  in-flvector))

(provide curve-iterations
         curve-parameters)

(define grid-size 300)
(define most-iterations 100)
(define step-tolerance 1e-15)
(define initial-damping 1e-3)

(define grid
  (for/flvector #:length grid-size ([i (in-range grid-size)])
    (fl/ (fl* 3.0 (exact->inexact i)) (exact->inexact (sub1 grid-size)))))

(define (targets r)
  (for/flvector #:length grid-size ([t (in-flvector grid)])
    (if (fl>= t r) (flexp (fl- r t)) 1.0)))

(define (power t b)
  (if (fl<= t 0.0) 0.0 (flexpt t (fl* 2.0 b))))

(define (residual-norm ys a b)
  (for/fold ([sum 0.0]) ([t (in-flvector grid)] [y (in-flvector ys)])
    (define e (fl- (fl/ 1.0 (fl+ 1.0 (fl* a (power t b)))) y))
    (fl+ sum (fl* e e))))

(define (normal-equations ys a b)
  (for/fold ([aa 0.0] [ab 0.0] [bb 0.0] [ga 0.0] [gb 0.0])
            ([t (in-flvector grid)] [y (in-flvector ys)])
    (define p (power t b))
    (define q (fl+ 1.0 (fl* a p)))
    (define e (fl- (fl/ 1.0 q) y))
    (define ja (fl/ (fl- 0.0 p) (fl* q q)))
    (define jb (if (fl<= t 0.0) 0.0 (fl* (fl* 2.0 (fllog t)) (fl* a ja))))
    (values (fl+ aa (fl* ja ja)) (fl+ ab (fl* ja jb)) (fl+ bb (fl* jb jb))
            (fl+ ga (fl* ja e)) (fl+ gb (fl* jb e)))))

(define (converged? step value)
  (fl<= (flabs step) (fl* step-tolerance (flabs value))))

(define (fit-normalised r)
  (define ys (targets r))
  (let loop ([a 1.0] [b 1.0] [damping initial-damping] [cost (residual-norm ys 1.0 1.0)] [k 0])
    (cond
      [(= k most-iterations) (values a b k)]
      [else
       (define-values (aa ab bb ga gb) (normal-equations ys a b))
       (define daa (fl+ aa damping))
       (define dbb (fl+ bb damping))
       (define det (fl- (fl* daa dbb) (fl* ab ab)))
       (define step-a (fl/ (fl- (fl* ab gb) (fl* dbb ga)) det))
       (define step-b (fl/ (fl- (fl* ab ga) (fl* daa gb)) det))
       (define next-a (fl+ a step-a))
       (define next-b (fl+ b step-b))
       (define next-cost (residual-norm ys next-a next-b))
       (cond
         [(and (converged? step-a a) (converged? step-b b)) (values a b (add1 k))]
         [(< next-cost cost) (loop next-a next-b (fl/ damping 10.0) next-cost (add1 k))]
         [else (loop a b (fl* damping 10.0) cost (add1 k))])])))

(define (curve-parameters min-dist spread)
  (define s (exact->inexact spread))
  (define-values (a b _) (fit-normalised (fl/ (exact->inexact min-dist) s)))
  (values (fl* a (flexpt s (fl* -2.0 b))) b))

(define (curve-iterations min-dist spread)
  (define-values (_a _b k) (fit-normalised (fl/ (exact->inexact min-dist) (exact->inexact spread))))
  k)
