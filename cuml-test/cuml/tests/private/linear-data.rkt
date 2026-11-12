#lang racket/base

(require (for-syntax racket/base)
         (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in math/flonum flstep)
         (only-in racket/flonum flsingle)
         (only-in racket/list drop first make-list take)
         (only-in racket/match match match-define struct*)
         (only-in racket/math exact-round sqr)
         (only-in rackunit check-true current-test-name test-case)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         (only-in threading ~>)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "exact-algebra.rkt"
                  condition
                  dot
                  eigen-bracket
                  integer-scaled
                  min-norm-solve
                  varying-inner))

(provide (struct-out lcase)
         (struct-out ridge-solution)
         (struct-out solution)
         (struct-out staging)
         algorithms
         case-fit
         case-result
         check-law
         check-law-case
         columns-of
         decimals
         docs-case
         dtypes
         dyadic
         exact-fit
         fit-gap
         fit-of
         fit-size
         floor-log2
         gap
         host-cosine
         insert-column
         kept-rows
         kappa-of
         law-case
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
         ridge-solve
         rounded
         rounded-case
         scaled-case
         scaled-columns
         staging-of
         unit-roundoff
         weighted-gram
         weighted-spread
         weights-or-ones
         zeroed)

(struct lcase (name X y w) #:transparent)
(struct solution (coef intercept kappa) #:transparent)
(struct staging (scale y-scale kappa) #:transparent)
(struct ridge-solution (alpha coef intercept kappa kappa₀ top bottom xty) #:transparent)

(define algorithms '(auto svd qr svd-qr))
(define dtypes '(float32 float64))

(define (unit-roundoff dtype)
  (if (eq? dtype 'float32)
      (expt 2.0 -24)
      (expt 2.0 -53)))

(define docs-case (lcase "docs" '((1 1) (1 2) (2 2) (2 3)) '(6 8 9 11) #f))

(define (decimals k cs)
  (map (lambda (c) (real->decimal-string c k)) cs))

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

(define (scaled-case c #:X [x-factor 1] #:y [y-factor 1] #:w [w-factor 1])
  (define (by f xs)
    (and xs (map (lambda (v) (* f v)) xs)))
  (lcase (lcase-name c)
         (map (lambda (row) (by x-factor row)) (lcase-X c))
         (by y-factor (lcase-y c))
         (by w-factor (lcase-w c))))

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

(define (floor-log2 x)
  (let loop ([e (- (integer-length (numerator x)) (integer-length (denominator x)))])
    (cond
      [(> (expt 2 e) x) (loop (sub1 e))]
      [(<= (expt 2 (add1 e)) x) (loop (add1 e))]
      [else e])))

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

(define (law-name name)
  (define outer (current-test-name))
  (if outer
      (format "~a ~a" outer name)
      name))

(define-syntax-parse-rule (law-case name:expr body:expr ...+)
  (test-case (law-name name)
    body ...))

(define-syntax-parse-rule (check-law-case name:expr got:expr bound:expr)
  (let ([law name])
    (law-case law
      (check-law law got bound))))

(define (columns-of rows)
  (apply map list rows))

(define (max-abs xs)
  (for/fold ([m 0]) ([x (in-list xs)])
    (max m (abs x))))

(define (weighted-gram X y w fit-intercept)
  (define weights (weights-or-ones w y))
  (define total (apply + weights))
  (match-define (cons dw ws) (integer-scaled weights))
  (define columns (map integer-scaled (columns-of X)))
  (define ys (integer-scaled y))
  (define (moment u v)
    (match-define (cons du us) u)
    (match-define (cons dv vs) v)
    (/ (for/sum ([a (in-list us)] [b (in-list vs)] [c (in-list ws)]) (* c a b)) (* dw du dv)))
  (define (first-moment u)
    (match-define (cons du us) u)
    (/ (for/sum ([a (in-list us)] [c (in-list ws)]) (* c a)) (* dw du)))
  (define sums (map first-moment columns))
  (define sy (first-moment ys))
  (define (centred su sv m)
    (if fit-intercept
        (- m (/ (* su sv) total))
        m))
  (define (mean s)
    (if fit-intercept
        (/ s total)
        0))
  (values (for/list ([u (in-list columns)]
                     [su (in-list sums)])
            (for/list ([v (in-list columns)]
                       [sv (in-list sums)])
              (centred su sv (moment u v))))
          (for/list ([u (in-list columns)]
                     [su (in-list sums)])
            (centred su sy (moment u ys)))
          (map mean sums)
          (mean sy)))

(define (oracle-lstsq X y w fit-intercept)
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define c (min-norm-solve G b))
  (solution c (- my (dot mx c)) (condition G)))

(define (staging-of X y w fit-intercept)
  (define-values (G _b _mx _my) (weighted-gram X y w fit-intercept))
  (define weights (weights-or-ones w y))
  (define exponents
    (for/list ([row (in-list G)]
               [j (in-naturals)])
      (define g (list-ref row j))
      (and (positive? g) (- (floor (/ (floor-log2 g) 2))))))
  (define kept (filter values exponents))
  (define rows (length (filter positive? weights)))
  (define shared (and (> (length kept) (- rows (if fit-intercept 1 0))) (apply min kept)))
  (define scale
    (for/list ([e (in-list exponents)])
      (expt 2 (or shared e 0))))
  (define staged
    (for/list ([row (in-list G)]
               [di (in-list scale)]
               [ei (in-list exponents)]
               #:when ei)
      (for/list ([g (in-list row)]
                 [dj (in-list scale)]
                 [ej (in-list exponents)]
                 #:when ej)
        (* di g dj))))
  (define largest
    (for/fold ([m 0])
              ([yi (in-list y)]
               [wi (in-list weights)]
               #:when (positive? wi))
      (max m (abs yi))))
  (staging scale
           (if (zero? largest)
               1
               (expt 2 (- (floor-log2 largest))))
           (if (null? staged)
               1
               (condition staged))))

(define (ridge-solve G b mx my inner outer alpha)
  (match-define (cons top bottom) outer)
  (define c
    (min-norm-solve (for/list ([row (in-list G)]
                               [i (in-naturals)])
                      (for/list ([g (in-list row)]
                                 [j (in-naturals)])
                        (if (= i j)
                            (+ g alpha)
                            g)))
                    b))
  (ridge-solution alpha
                  c
                  (- my (dot mx c))
                  (/ (+ top alpha) (+ bottom alpha))
                  (match inner
                    [(cons _ 0) +inf.0]
                    [(cons high low) (/ high low)])
                  top
                  bottom
                  (sqrt (dot b b))))

(define (oracle-ridge X y w alpha fit-intercept)
  (first (ridge-oracles X y w fit-intercept (list alpha) #:relative? #f)))

(define (ridge-oracles X y w fit-intercept ratios #:relative? [relative? #t])
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define-values (inner outer) (eigen-bracket G))
  (define inner₀ (varying-inner G inner))
  (match-define (cons top _) outer)
  (define unit
    (if relative?
        (expt 2 (exact-round (/ (log top) (log 2))))
        1))
  (for/list ([r (in-list ratios)])
    (ridge-solve G b mx my inner₀ outer (* r unit))))

(define (host-cosine X y w coef intercept fit-intercept #:alpha [alpha 0])
  (match-define (cons sw W) (integer-scaled (weights-or-ones w y)))
  (define W-total (apply + W))
  (define (centred-product xs vs)
    (match-define (cons su U) xs)
    (match-define (cons sv V) vs)
    (define raw (for/sum ([u (in-list U)] [v (in-list V)] [wi (in-list W)]) (* wi u v)))
    (/ (if fit-intercept
           (- raw (/ (* (dot W U) (dot W V)) W-total))
           raw)
       (* sw su sv)))
  (define r
    (integer-scaled (for/list ([row (in-list X)]
                               [yi (in-list y)])
                      (- yi intercept (dot row coef)))))
  (define y-norm (sqrt (centred-product (integer-scaled y) (integer-scaled y))))
  (define widest (* alpha (max-abs coef)))
  (for/fold ([worst 0.0])
            ([col (in-list (columns-of X))]
             [cj (in-list coef)])
    (define xc (integer-scaled col))
    (define num (- (centred-product xc r) (* alpha cj)))
    (define square (centred-product xc xc))
    (if (zero? square)
        worst
        (max worst (exact->inexact (/ (abs num) (+ (* (sqrt square) y-norm) widest)))))))

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
  (check-equal? b3 2)
  (check-equal? (staging-of '((1 1000) (2 3000) (4 2000) (3 5000)) '(1 3 2 4) #f #t)
                (staging (list 1/2 (expt 2 -11))
                         1/4
                         (condition (list (list 5/4 (* 2500 (expt 2 -12)))
                                          (list (* 2500 (expt 2 -12)) (* 8750000 (expt 2 -22)))))))
  (match-define (staging wide-scale _ _) (staging-of '((1 64) (2 0)) '(1 2) #f #t))
  (check-equal? wide-scale (list 1/32 1/32))
  (match-define (staging constant-scale _ constant-kappa)
    (staging-of '((1 5) (2 5) (4 5)) '(1 2 3) #f #t))
  (check-equal? constant-scale (list 1/2 1))
  (check-true (<= 1 constant-kappa 2) (format "κ over the varying column ~a" constant-kappa))
  (match-define (list (struct* ridge-solution ([kappa₀ constant-kappa₀])))
    (ridge-oracles '((1 5) (2 5) (4 5)) '(1 2 3) #f #t '(1)))
  (check-true (<= 1 constant-kappa₀ 2) (format "κ₀ over the varying column ~a" constant-kappa₀)))
