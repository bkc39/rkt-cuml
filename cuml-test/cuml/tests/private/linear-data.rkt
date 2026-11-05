#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in math/flonum flstep)
         (only-in racket/flonum flsingle)
         (only-in racket/list drop last make-list take)
         (only-in racket/match match match-define)
         (only-in racket/math exact-round sqr)
         (only-in rackunit check-true)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         (only-in threading ~>)
         (only-in "blob-data.rkt" normal-source uniform-source))

(provide (struct-out lcase)
         (struct-out ridge-solution)
         (struct-out solution)
         algorithms
         case-fit
         case-result
         check-law
         columns-of
         dtypes
         dyadic
         exact-fit
         fit-gap
         fit-of
         fit-size
         gap
         host-cosine
         insert-column
         kept-rows
         kappa-of
         law-cases
         reach-fixtures
         max-abs
         normal-case
         on-matrix
         on-vector
         oracle-lstsq
         oracle-ridge
         ridge-oracles
         power-spread-case
         power-spread-gone
         remove-at
         result-fit
         rounded
         rounded-case
         scaled-columns
         unit-roundoff
         weights-or-ones
         zeroed)

(struct lcase (name X y w) #:transparent)
(struct solution (coef intercept kappa) #:transparent)
(struct ridge-solution (alpha coef intercept kappa top bottom xty) #:transparent)

(define algorithms '(auto svd qr svd-qr))
(define dtypes '(float32 float64))

(define (unit-roundoff dtype)
  (if (eq? dtype 'float32)
      (expt 2.0 -24)
      (expt 2.0 -53)))

(define (dyadic x [bits 10])
  (/ (round (* (inexact->exact x) (expt 2 bits))) (expt 2 bits)))

(define (normal-case n d seed #:weighted? [weighted? #f] #:noise [noise 1/10])
  (define next (normal-source (+ 1000 seed (* 7 n) (* 131 d))))
  (define draw (uniform-source (+ 5000 seed)))
  (define beta
    (for/list ([_ (in-range d)])
      (dyadic (next))))
  (define X
    (for/list ([_ (in-range n)])
      (for/list ([_ (in-range d)])
        (dyadic (next)))))
  (define y
    (for/list ([row (in-list X)])
      (dyadic (+ 3 (dot row beta) (* noise (next))))))
  (define w
    (and weighted?
         (for/list ([_ (in-range n)])
           (dyadic (+ 1/2 (* 3/2 (draw))) 6))))
  (lcase (format "normal ~ax~a seed ~a~a" n d seed (if weighted? " weighted" "")) X y w))

(define law-shapes
  '((20 2 (0 1)) (20 7 (0 1)) (200 2 (0 1)) (200 7 (0 1)) (200 30 (0 1)) (5000 7 (0))))

(define (law-cases)
  (append (for*/list ([shape (in-list law-shapes)]
                      #:do [(match-define (list n d seeds) shape)]
                      [seed (in-list seeds)]
                      [weighted? (in-list '(#f #t))])
            (normal-case n d seed #:weighted? weighted?))
          (for/list ([seed (in-range 2 6)])
            (normal-case 200 7 seed #:weighted? (odd? seed)))))

(define (on-matrix rows [dtype 'float64])
  (list*->device-matrix rows #:dtype dtype))

(define (on-vector xs [dtype 'float64])
  (and xs (list->device-vector xs #:dtype dtype)))

(define (case-result c dt #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (linear-regression-fit (on-matrix (lcase-X c) dt)
                         (on-vector (lcase-y c) dt)
                         #:sample-weight (on-vector (lcase-w c) dt)
                         #:algorithm a
                         #:fit-intercept fi))

(define (fit-of X y w dt #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (case-fit (lcase "" X y w) dt #:algorithm a #:fit-intercept fi))

(define (case-fit c dt #:algorithm [a 'auto] #:fit-intercept [fi #t])
  (result-fit (case-result c dt #:algorithm a #:fit-intercept fi)))

(define (result-fit r)
  (list (~> r linear-regression-result-coef device-vector->list)
        (linear-regression-result-intercept r)))

(define (exact-fit f)
  (match-define (list coef b) f)
  (list (map inexact->exact coef) (inexact->exact b)))

(define (fit-gap f g)
  (match-define (list coef b) (exact-fit f))
  (match-define (list coef* b*) (exact-fit g))
  (gap (cons b coef) (cons b* coef*)))

(define (fit-size f)
  (match-define (list coef b) (exact-fit f))
  (max 1 (abs b) (max-abs coef)))

(define (gap xs ys)
  (max-abs (map - xs ys)))

(define (kappa-of c fi)
  (solution-kappa (oracle-lstsq (lcase-X c) (lcase-y c) (lcase-w c) fi)))

(define (scaled-columns c factors y-factor)
  (lcase (lcase-name c)
         (for/list ([row (in-list (lcase-X c))])
           (for/list ([x (in-list row)]
                      [s (in-list factors)])
             (* x s)))
         (map (lambda (v) (* v y-factor)) (lcase-y c))
         (lcase-w c)))

(define (reach-fixtures dt)
  (define column-scales
    (if (eq? dt 'float32)
        '(#e1e-5 #e1e-7 #e1e30)
        '(#e1e150 #e1e-150)))
  (append (for/list ([s (in-list column-scales)])
            (list (format "column 1 at ~a" s) (list 1 s 1 1 1 1 1) 1))
          (list (list "X at 1e-12" (make-list 7 #e1e-12) 1)
                (list "y at 1e37" (make-list 7 1) #e1e37)
                (list "subnormal y" (make-list 7 1) #e1e-40))))

(define power-spread-gone '(3 50 101))

(define (kept-rows xs gone)
  (for/list ([x (in-list xs)]
             [i (in-naturals)]
             #:unless (memv i gone))
    x))

(define (zeroed ws gone [to 0])
  (for/list ([v (in-list ws)]
             [i (in-naturals)])
    (if (memv i gone) to v)))

(define (weights-or-ones w y)
  (or w (make-list (length y) 1)))

(define (insert-column rows at value-of)
  (for/list ([row (in-list rows)]
             [i (in-naturals)])
    (append (take row at) (list (value-of i)) (drop row at))))

(define (remove-at xs at)
  (append (take xs at) (drop xs (add1 at))))

(define (weighted-spread xs ws)
  (define m (/ (for/sum ([x (in-list xs)] [w (in-list ws)]) (* w x)) (apply + ws)))
  (for/sum ([x (in-list xs)] [w (in-list ws)]) (* w (sqr (- x m)))))

(define (power-spread-case seed step)
  (define n 200)
  (define draw (uniform-source (+ 101 seed)))
  (define (draws f)
    (for/list ([_ (in-range n)])
      (f (draw))))
  (define w (draws (lambda (u) (+ 0.5 (* 1.5 u)))))
  (define x2 (draws (lambda (u) (- (* 2 u) 1))))
  (define raw (draws (lambda (u) (- (* 2 u) 1))))
  (define y
    (for/list ([a (in-list raw)]
               [b (in-list x2)])
      (+ (* 3 a) (* -2 b) 0.5 (* 0.01 (draw)))))
  (define (spread xs)
    (weighted-spread (map inexact->exact (kept-rows xs power-spread-gone))
                     (map inexact->exact (kept-rows w power-spread-gone))))
  (define c (/ 1.0 (sqrt (exact->inexact (spread raw)))))
  (define x1 (map (lambda (v) (* c v)) raw))
  (define-values (j xj)
    (for/fold ([j 0]
               [xj 0.0])
              ([x (in-list x1)]
               [i (in-naturals)]
               #:unless (memv i power-spread-gone))
      (if (> (abs x) (abs xj))
          (values i x)
          (values j xj))))
  (define (stepped k)
    (for/list ([x (in-list x1)]
               [i (in-naturals)])
      (if (= i j)
          (flstep xj k)
          x)))
  (define per (- (spread (stepped 1)) (spread (stepped 0))))
  (define k0 (exact-round (/ (- 1 (spread (stepped 0))) per)))
  (lcase (format "spread at 1, seed ~a step ~a" seed step) (map list (stepped (+ k0 step)) x2) y w))

(define (rounded dt x)
  (if (eq? dt 'float32)
      (flsingle (real->double-flonum x))
      (real->double-flonum x)))

(define (rounded-case c dt)
  (define (each xs)
    (and xs (map (lambda (x) (inexact->exact (rounded dt x))) xs)))
  (lcase (lcase-name c) (map each (lcase-X c)) (each (lcase-y c)) (each (lcase-w c))))

(define (within got bound)
  (if (zero? bound)
      (if (zero? got) 0.0 +inf.0)
      (exact->inexact (/ got bound))))

(define (check-law name got bound)
  (define ratio (within got bound))
  (check-true (<= ratio 1) (format "~a: ~a of the bound" name ratio)))

(define (columns-of rows)
  (apply map list rows))

(define (max-abs xs)
  (for/fold ([m 0]) ([x (in-list xs)])
    (max m (abs x))))

(define (dot u v)
  (for/sum ([a (in-list u)] [b (in-list v)]) (* a b)))

(define (mat*vec A v)
  (for/list ([row (in-list A)])
    (dot row v)))

(define (weighted-gram X y w fit-intercept)
  (define weights (weights-or-ones w y))
  (define total (apply + weights))
  (define (mean xs)
    (if fit-intercept
        (/ (dot weights xs) total)
        0))
  (define (centred xs)
    (define m (mean xs))
    (map (lambda (x) (- x m)) xs))
  (define columns (map centred (columns-of X)))
  (define yc (centred y))
  (define (weighted-dot u v)
    (for/sum ([a (in-list u)] [b (in-list v)] [wi (in-list weights)]) (* wi a b)))
  (values (for/list ([u (in-list columns)])
            (for/list ([v (in-list columns)])
              (weighted-dot u v)))
          (for/list ([u (in-list columns)])
            (weighted-dot u yc))
          (map mean (columns-of X))
          (mean y)))

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

(define (min-norm-solve A b)
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

(define (oracle-lstsq X y w fit-intercept)
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define c (min-norm-solve G b))
  (solution c (- my (dot mx c)) (condition G)))

(define (ridge-solve G b mx my top bottom alpha)
  (define c
    (min-norm-solve (for/list ([row (in-list G)]
                               [i (in-naturals)])
                      (for/list ([g (in-list row)]
                                 [j (in-naturals)])
                        (if (= i j) (+ g alpha) g)))
                    b))
  (ridge-solution alpha c (- my (dot mx c)) (/ (+ top alpha) (+ bottom alpha)) top bottom (sqrt (dot b b))))

(define (oracle-ridge X y w alpha fit-intercept)
  (car (ridge-oracles X y w fit-intercept (list alpha) #:relative? #f)))

(define (ridge-oracles X y w fit-intercept ratios #:relative? [relative? #t])
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define-values (top bottom) (eigen-range G))
  (define unit (if relative? (expt 2 (exact-round (/ (log top) (log 2)))) 1))
  (for/list ([r (in-list ratios)])
    (ridge-solve G b mx my top bottom (* r unit))))

(define (host-cosine X y w coef intercept fit-intercept)
  (define weights (weights-or-ones w y))
  (define total (apply + weights))
  (define (centre xs)
    (define m
      (if fit-intercept
          (/ (dot weights xs) total)
          0))
    (map (lambda (x) (- x m)) xs))
  (define r
    (for/list ([row (in-list X)]
               [yi (in-list y)])
      (- yi intercept (dot row coef))))
  (define yc (centre y))
  (define y-norm (sqrt (for/sum ([v (in-list yc)] [wi (in-list weights)]) (* wi v v))))
  (for/fold ([worst 0.0]) ([col (in-list (columns-of X))])
    (define xc (centre col))
    (define num (for/sum ([x (in-list xc)] [ri (in-list r)] [wi (in-list weights)]) (* wi x ri)))
    (define norm (sqrt (for/sum ([x (in-list xc)] [wi (in-list weights)]) (* wi x x))))
    (max worst (exact->inexact (/ (abs num) (* norm y-norm))))))

(module+ test
  (require (only-in rackunit check-equal? check-true))
  (match-define (solution c b k) (oracle-lstsq '((1 1) (1 2) (2 2) (2 3)) '(6 8 9 11) #f #t))
  (check-equal? c '(1 2))
  (check-equal? b 3)
  (check-true (< 1 k 10))
  (match-define (solution c2 b2 _) (oracle-lstsq '((1 2) (2 4) (3 6)) '(1 2 3) #f #f))
  (check-equal? c2 '(1/5 2/5))
  (check-equal? b2 0)
  (match-define (solution c3 b3 _) (oracle-lstsq '((1 0 0) (0 1 0) (0 0 1)) '(1 2 3) #f #t))
  (check-equal? c3 '(-1 0 1))
  (check-equal? b3 2))
