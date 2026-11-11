#lang racket/base

(require (only-in racket/list last make-list splitf-at)
         (only-in racket/match match match-define))

(provide condition
         dot
         eigen-range
         mat*vec
         min-norm-solve
         rayleigh-range)

(define (dot u v)
  (for/sum ([a (in-list u)] [b (in-list v)]) (* a b)))

(define (mat*vec A v)
  (for/list ([row (in-list A)])
    (dot row v)))

(define (swapped M r p)
  (define row-r (list-ref M r))
  (define row-p (list-ref M p))
  (for/list ([row (in-list M)]
             [i (in-naturals)])
    (cond
      [(= i r) row-p]
      [(= i p) row-r]
      [else row])))

(define (pivoted M r c)
  (define row (list-ref M r))
  (define pivot (map (lambda (x) (/ x (list-ref row c))) row))
  (for/list ([row (in-list M)]
             [i (in-naturals)])
    (define f (list-ref row c))
    (cond
      [(= i r) pivot]
      [(zero? f) row]
      [else (map (lambda (x p) (- x (* f p))) row pivot)])))

(define (pivot-row M r c)
  (for/first ([row (in-list (list-tail M r))]
              [i (in-naturals r)]
              #:unless (zero? (list-ref row c)))
    i))

(define (rref A b)
  (define d (length A))
  (let loop ([M (map (lambda (row bi) (append row (list bi))) A b)]
             [r 0]
             [c 0]
             [pivots '()])
    (cond
      [(or (= r d) (= c d)) (values M (reverse pivots))]
      [(pivot-row M r c)
       =>
       (lambda (p) (loop (pivoted (swapped M r p) r c) (add1 r) (add1 c) (cons c pivots)))]
      [else (loop M r (add1 c) pivots)])))

(define (scaled-to-integers rows)
  (define d
    (for*/fold ([d 1])
               ([row (in-list rows)]
                [x (in-list row)])
      (lcm d (denominator x))))
  (for/list ([row (in-list rows)])
    (map (lambda (x) (* d x)) row)))

(define (pivot-first M)
  (define-values (zeros rest) (splitf-at M (lambda (row) (zero? (car row)))))
  (and (pair? rest) (cons (car rest) (append zeros (cdr rest)))))

(define (bareiss M previous)
  (match (and (pair? M) (pivot-first M))
    [#f (and (null? M) '())]
    [(cons (and top (cons pivot pivot-tail)) rest)
     (define below
       (bareiss (for/list ([row (in-list rest)])
                  (match-define (cons f tail) row)
                  (map (lambda (x p) (quotient (- (* pivot x) (* f p)) previous)) tail pivot-tail))
                pivot))
     (and below (cons top below))]))

(define (back-substituted rows)
  (for/foldr ([x '()])
             ([row (in-list rows)])
             (match-define (cons pivot tail) row)
             (cons (/ (- (last tail) (dot tail x)) pivot) x)))

(define (nonsingular-solve A b)
  (define M (map (lambda (row bi) (append row (list bi))) A b))
  (define rows
    (and (for*/and ([row (in-list M)]
                    [x (in-list row)])
           (exact? x))
         (bareiss (scaled-to-integers M) 1)))
  (and rows (back-substituted rows)))

(define (min-norm-solve A b)
  (or (nonsingular-solve A b) (rref-min-norm-solve A b)))

(define (rref-min-norm-solve A b)
  (define d (length A))
  (define-values (M pivots) (rref A b))
  (define pivot-rows
    (for/hash ([c (in-list pivots)]
               [row (in-list M)])
      (values c row)))
  (define p
    (for/list ([c (in-range d)])
      (match (hash-ref pivot-rows c #f)
        [#f 0]
        [row (last row)])))
  (define N
    (for/list ([f (in-range d)]
               #:unless (hash-has-key? pivot-rows f))
      (for/list ([c (in-range d)])
        (match (hash-ref pivot-rows c #f)
          [_
           #:when (= c f)
           1]
          [#f 0]
          [row (- (list-ref row f))]))))
  (match N
    ['() p]
    [_
     (define alpha
       (min-norm-solve (for/list ([u (in-list N)])
                         (for/list ([v (in-list N)])
                           (dot u v)))
                       (for/list ([u (in-list N)])
                         (dot u p))))
     (for/fold ([x p])
               ([a (in-list alpha)]
                [u (in-list N)])
       (map (lambda (xj uj) (- xj (* a uj))) x u))]))

(define (power-top G steps)
  (define d (length G))
  (for/fold ([v (make-list d 1.0)]
             [lam 0.0]
             #:result lam)
            ([_ (in-range steps)])
    (define Gv (mat*vec G v))
    (define norm (sqrt (dot Gv Gv)))
    (if (zero? norm)
        (values v 0.0)
        (values (map (lambda (x) (/ x norm)) Gv) norm))))

(define (eigen-range G-exact)
  (define G
    (for/list ([row (in-list G-exact)])
      (map exact->inexact row)))
  (define top (power-top G 300))
  (define shifted
    (for/list ([row (in-list G)]
               [i (in-naturals)])
      (for/list ([g (in-list row)]
                 [j (in-naturals)])
        (- (if (= i j) top 0.0) g))))
  (values top (- top (power-top shifted 600))))

(define (condition G-exact)
  (define-values (top bottom) (eigen-range G-exact))
  (if (> bottom (* 1e-14 top))
      (sqrt (/ top bottom))
      +inf.0))

(define (unit v)
  (define norm (sqrt (dot v v)))
  (map (lambda (x) (/ x norm)) v))

(define (rayleigh G v)
  (define e (map inexact->exact v))
  (/ (dot e (mat*vec G e)) (dot e e)))

(define (rayleigh-range G)
  (define G-inexact
    (for/list ([row (in-list G)])
      (map exact->inexact row)))
  (define ones (make-list (length G) 1.0))
  (define top-vector
    (for/fold ([v ones]) ([_ (in-range 300)])
      (unit (mat*vec G-inexact v))))
  (define bottom-vector
    (for/fold ([v ones]) ([_ (in-range 3)])
      (unit (map exact->inexact (min-norm-solve G (map inexact->exact v))))))
  (values (rayleigh G top-vector) (rayleigh G bottom-vector)))

(module+ test
  (require (only-in rackunit check-equal? check-true))
  (define A '((2 1 0) (1 3 1) (0 1 4)))
  (define b '(1 2 3))
  (check-equal? (min-norm-solve A b) (rref-min-norm-solve A b))
  (check-equal? (min-norm-solve '((0 1) (1 0)) '(2 3)) '(3 2))
  (check-equal? (min-norm-solve '((1 2) (2 4)) '(1 2)) '(1/5 2/5))
  (check-equal? (nonsingular-solve '((1 2) (2 4)) '(1 2)) #f)
  (check-equal? (nonsingular-solve '((1.0 0) (0 1)) '(1 2)) #f)
  (check-equal? (min-norm-solve '((1/3 1/7) (1/7 1/5)) '(1/2 1/9))
                (rref-min-norm-solve '((1/3 1/7) (1/7 1/5)) '(1/2 1/9)))
  (define-values (top bottom) (rayleigh-range '((4 0 0) (0 2 0) (0 0 1/1000))))
  (check-true (and (exact? top) (exact? bottom)))
  (check-true (<= (* 999/1000 4) top 4))
  (check-true (<= 1/1000 bottom (* 1001/1000 1/1000))))
