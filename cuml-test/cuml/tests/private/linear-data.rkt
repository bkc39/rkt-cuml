#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-result-coef
                  linear-regression-result-intercept)
         (only-in racket/flonum flsingle)
         (only-in racket/list make-list)
         (only-in raft device-vector->list list*->device-matrix list->device-vector)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "estimator-data.rkt" apply-options))

(provide (struct-out lcase)
         (struct-out solution)
         algorithms
         columns-of
         dtypes
         dyadic
         fitted-of
         host-cosine
         law-cases
         lcase-device
         max-abs
         normal-case
         on-matrix
         on-vector
         oracle-lstsq
         rounded-case
         unit-roundoff
         within)

(struct lcase (name X y w) #:transparent)
(struct solution (coef intercept kappa) #:transparent)

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
      (dyadic (+ 3 (for/sum ([x (in-list row)] [b (in-list beta)]) (* x b)) (* noise (next))))))
  (define w
    (and weighted?
         (for/list ([_ (in-range n)])
           (dyadic (+ 1/2 (* 3/2 (draw))) 6))))
  (lcase (format "normal ~ax~a seed ~a~a" n d seed (if weighted? " weighted" "")) X y w))

(define (law-cases)
  (append (for*/list ([nd (in-list '((20 2) (20 7) (200 2) (200 7) (200 30) (5000 7)))]
                      [seed (in-list (if (= (car nd) 5000) '(0) '(0 1)))]
                      [weighted? (in-list '(#f #t))])
            (normal-case (car nd) (cadr nd) seed #:weighted? weighted?))
          (for/list ([seed (in-range 2 6)])
            (normal-case 200 7 seed #:weighted? (odd? seed)))))

(define (on-matrix rows dtype)
  (list*->device-matrix rows #:dtype dtype))

(define (on-vector xs dtype)
  (and xs (list->device-vector xs #:dtype dtype)))

(define (lcase-device c dtype)
  (values (on-matrix (lcase-X c) dtype) (on-vector (lcase-y c) dtype) (on-vector (lcase-w c) dtype)))

(define (fitted-of c dtype options)
  (define-values (X y w) (lcase-device c dtype))
  (define r (apply-options linear-regression-fit (hash-set options 'sample-weight w) X y))
  (values (device-vector->list (linear-regression-result-coef r))
          (linear-regression-result-intercept r)))

(define (rounded dt x)
  (inexact->exact (if (eq? dt 'float32)
                      (flsingle (real->double-flonum x))
                      (real->double-flonum x))))

(define (rounded-case c dt)
  (define (each xs)
    (and xs (map (lambda (x) (rounded dt x)) xs)))
  (lcase (lcase-name c) (map each (lcase-X c)) (each (lcase-y c)) (each (lcase-w c))))

(define (within got bound)
  (if (zero? bound)
      (if (zero? got) 0.0 +inf.0)
      (exact->inexact (/ got bound))))

(define (columns-of rows)
  (apply map list rows))

(define (max-abs xs)
  (for/fold ([m 0]) ([x (in-list xs)])
    (max m (abs x))))

(define (mat*vec A v)
  (for/list ([row (in-list A)])
    (for/sum ([a (in-list row)] [x (in-list v)]) (* a x))))

(define (weighted-gram X y w fit-intercept)
  (define weights (or w (make-list (length y) 1)))
  (define total (apply + weights))
  (define d (length (car X)))
  (define sx
    (for/fold ([acc (make-list d 0)]) ([row (in-list X)] [wi (in-list weights)])
      (map (lambda (a x) (+ a (* wi x))) acc row)))
  (define sy (for/sum ([yi (in-list y)] [wi (in-list weights)]) (* wi yi)))
  (define mx (if fit-intercept (map (lambda (s) (/ s total)) sx) (make-list d 0)))
  (define my (if fit-intercept (/ sy total) 0))
  (define G (for/vector ([_ (in-range d)]) (make-vector d 0)))
  (define b (make-vector d 0))
  (for ([row (in-list X)] [yi (in-list y)] [wi (in-list weights)])
    (define xc (for/vector ([x (in-list row)] [m (in-list mx)]) (- x m)))
    (define yc (- yi my))
    (for ([j (in-range d)])
      (define wx (* wi (vector-ref xc j)))
      (vector-set! b j (+ (vector-ref b j) (* wx yc)))
      (define Gj (vector-ref G j))
      (for ([k (in-range j d)])
        (vector-set! Gj k (+ (vector-ref Gj k) (* wx (vector-ref xc k)))))))
  (for* ([j (in-range d)] [k (in-range j)])
    (vector-set! (vector-ref G j) k (vector-ref (vector-ref G k) j)))
  (values (for/list ([row (in-vector G)]) (vector->list row)) (vector->list b) mx my))

(define (rref A b)
  (define d (length A))
  (define M (for/vector ([row (in-list A)] [bi (in-list b)]) (list->vector (append row (list bi)))))
  (define pivots
    (let loop ([r 0] [c 0] [pivots '()])
      (cond
        [(or (= r d) (= c d)) (reverse pivots)]
        [(for/first ([i (in-range r d)] #:unless (zero? (vector-ref (vector-ref M i) c))) i)
         => (lambda (p)
              (define tmp (vector-ref M p))
              (vector-set! M p (vector-ref M r))
              (vector-set! M r tmp)
              (define pr (vector-ref M r))
              (define pv (vector-ref pr c))
              (for ([k (in-range (add1 d))])
                (vector-set! pr k (/ (vector-ref pr k) pv)))
              (for ([i (in-range d)] #:unless (= i r))
                (define ri (vector-ref M i))
                (define f (vector-ref ri c))
                (unless (zero? f)
                  (for ([k (in-range (add1 d))])
                    (vector-set! ri k (- (vector-ref ri k) (* f (vector-ref pr k)))))))
              (loop (add1 r) (add1 c) (cons c pivots)))]
        [else (loop r (add1 c) pivots)])))
  (values M pivots))

(define (min-norm-solve A b)
  (define d (length A))
  (define-values (M pivots) (rref A b))
  (define particular (make-vector d 0))
  (for ([c (in-list pivots)] [r (in-naturals)])
    (vector-set! particular c (vector-ref (vector-ref M r) d)))
  (define free (for/list ([c (in-range d)] #:unless (memv c pivots)) c))
  (define null-basis
    (for/list ([f (in-list free)])
      (define v (make-vector d 0))
      (vector-set! v f 1)
      (for ([c (in-list pivots)] [r (in-naturals)])
        (vector-set! v c (- (vector-ref (vector-ref M r) f))))
      (vector->list v)))
  (define p (vector->list particular))
  (cond
    [(null? null-basis) p]
    [else
     (define N null-basis)
     (define NtN (for/list ([u (in-list N)]) (for/list ([v (in-list N)]) (dot u v))))
     (define Ntp (for/list ([u (in-list N)]) (dot u p)))
     (define alpha (min-norm-solve NtN Ntp))
     (for/list ([j (in-range d)] [pj (in-list p)])
       (- pj (for/sum ([a (in-list alpha)] [u (in-list N)]) (* a (list-ref u j)))))]))

(define (dot u v)
  (for/sum ([a (in-list u)] [b (in-list v)]) (* a b)))

(define (power-top G steps)
  (define d (length G))
  (for/fold ([v (make-list d 1.0)] [lam 0.0] #:result lam) ([_ (in-range steps)])
    (define Gv (mat*vec G v))
    (define norm (sqrt (dot Gv Gv)))
    (if (zero? norm)
        (values v 0.0)
        (values (map (lambda (x) (/ x norm)) Gv) norm))))

(define (condition G-exact)
  (define G (for/list ([row (in-list G-exact)]) (map exact->inexact row)))
  (define top (power-top G 300))
  (define shifted
    (for/list ([row (in-list G)] [i (in-naturals)])
      (for/list ([g (in-list row)] [j (in-naturals)])
        (- (if (= i j) top 0.0) g))))
  (define bottom (- top (power-top shifted 600)))
  (if (> bottom (* 1e-14 top))
      (sqrt (/ top bottom))
      +inf.0))

(define (oracle-lstsq X y w fit-intercept)
  (define-values (G b mx my) (weighted-gram X y w fit-intercept))
  (define c (min-norm-solve G b))
  (solution c (- my (dot mx c)) (condition G)))

(define (host-cosine X y w coef intercept fit-intercept)
  (define weights (or w (make-list (length y) 1)))
  (define total (apply + weights))
  (define (centre xs)
    (define m (if fit-intercept (/ (for/sum ([x (in-list xs)] [wi (in-list weights)]) (* wi x)) total) 0))
    (map (lambda (x) (- x m)) xs))
  (define r
    (for/list ([row (in-list X)] [yi (in-list y)])
      (- yi intercept (dot row coef))))
  (define yc (centre y))
  (define y-norm (sqrt (for/sum ([v (in-list yc)] [wi (in-list weights)]) (* wi v v))))
  (for/fold ([worst 0.0]) ([col (in-list (columns-of X))])
    (define xc (centre col))
    (define num (for/sum ([x (in-list xc)] [ri (in-list r)] [wi (in-list weights)]) (* wi x ri)))
    (define norm (sqrt (for/sum ([x (in-list xc)] [wi (in-list weights)]) (* wi x x))))
    (max worst (exact->inexact (/ (abs num) (* norm y-norm))))))

(module+ test
  (require (only-in racket/match match-define)
           (only-in rackunit check-equal? check-true))
  (match-define (solution c b k) (oracle-lstsq '((1 1) (1 2) (2 2) (2 3)) '(6 8 9 11) #f #t))
  (check-equal? c '(1 2))
  (check-equal? b 3)
  (check-true (< 1 k 10))
  (match-define (solution c2 b2 _) (oracle-lstsq '((1 2) (2 4) (3 6)) '(1 2 3) #f #f))
  (check-equal? c2 '(1/5 2/5))
  (check-equal? b2 0))
