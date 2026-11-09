#lang racket/base

(require (only-in cuml/linear-model
                  elastic-net-fit
                  elastic-net-result-coef
                  elastic-net-result-intercept
                  elastic-net-result-n-iter
                  lasso-result
                  lasso-result-coef
                  lasso-result-intercept
                  lasso-result-n-iter)
         (only-in racket/list append* make-list)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-true test-case)
         (only-in raft device-vector->list)
         (only-in threading ~>)
         (only-in "blob-data.rkt" normal-source)
         (only-in "elastic-net-oracle.rkt"
                  alpha-max
                  en-solution
                  host-kkt
                  kkt
                  oracle-elastic-net
                  problem-of
                  s2-bound)
         (only-in "linear-data.rkt"
                  check-law
                  dot
                  dtypes
                  dyadic
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  max-abs
                  on-matrix
                  on-vector
                  unit-roundoff))

(provide alpha-ratios
         check-elastic-net-laws
         check-s2
         correlated-case
         distance
         dyadic-weight-case
         en-case-fit
         en-fit
         l1-ratios
         law-alpha
         mirrored-weight-case
         norm2
         selections
         tols)

(define l1-ratios '(1 0.5 0.05 0))
(define alpha-ratios '(0.3 0.01 1e-4))
(define tols '(1e-2 1e-3 1e-5))
(define selections '(cyclic random))

(define (norm2 xs)
  (sqrt (for/sum ([x (in-list xs)]) (* x x))))

(define (correlated-case n d corr seed #:weighted? [weighted? #f])
  (define next (normal-source (+ 7000 seed (* 7 n) (* 131 d))))
  (define a (sqrt corr))
  (define b (sqrt (- 1 corr)))
  (define beta
    (for/list ([j (in-range d)])
      (if (zero? (modulo (+ j seed) 3))
          0
          (dyadic (next)))))
  (define X
    (for/list ([_ (in-range n)])
      (define shared (next))
      (for/list ([_ (in-range d)])
        (dyadic (+ (* a shared) (* b (next)))))))
  (define y
    (for/list ([row (in-list X)])
      (dyadic (+ 3 (dot row beta) (* 1/10 (next))))))
  (define w
    (and weighted?
         (for/list ([_ (in-range n)])
           (dyadic (+ 1/2 (* 1/4 (abs (next)))) 6))))
  (lcase (format "correlated ~ax~a corr ~a seed ~a~a" n d corr seed (if weighted? " weighted" ""))
         X
         y
         w))

(define dyadic-groups '((1 32) (4 24) (16 8)))

(define (dyadic-weights)
  (append* (for/list ([g (in-list dyadic-groups)])
             (match-define (list w k) g)
             (make-list k w))))

(define (dyadic-weight-case seed)
  (match-define (lcase _ X y _) (correlated-case 64 4 1/2 seed))
  (lcase (format "dyadic weights seed ~a" seed) X y (dyadic-weights)))

(define (mirrored-weight-case seed)
  (match-define (lcase _ X y _) (correlated-case 32 4 1/2 seed))
  (define w (dyadic-weights))
  (define (mirror rows)
    (for/fold ([out '()]
               #:result (reverse out))
              ([row (in-list rows)])
      (list* (map - row) row out)))
  (define pairs
    (for/fold ([out '()]
               #:result (reverse out))
              ([v (in-list y)])
      (list* (- v) v out)))
  (lcase (format "mirrored dyadic weights seed ~a" seed)
         (mirror X)
         pairs
         (for/list ([k (in-range 64)])
           (list-ref w (* 2 (quotient k 2))))))

(define (en-fit r)
  (match r
    [(lasso-result c b k) (list (device-vector->list c) b k)]
    [_
     (list (~> r elastic-net-result-coef device-vector->list)
           (elastic-net-result-intercept r)
           (elastic-net-result-n-iter r))]))

(define (en-case-fit c
                     dt
                     #:alpha alpha
                     #:l1-ratio [l1-ratio 0.5]
                     #:fit-intercept [fi #t]
                     #:max-iter [max-iter 1000]
                     #:tol [tol 1e-3]
                     #:selection [selection 'cyclic])
  (en-fit (elastic-net-fit (on-matrix (lcase-X c) dt)
                           (on-vector (lcase-y c) dt)
                           #:sample-weight (on-vector (lcase-w c) dt)
                           #:alpha alpha
                           #:l1-ratio l1-ratio
                           #:fit-intercept fi
                           #:max-iter max-iter
                           #:tol tol
                           #:selection selection)))

(define (law-alpha p ratio l1-ratio)
  (* ratio (exact->inexact (alpha-max p (if (zero? l1-ratio) 1 l1-ratio)))))

(define (rows-of c)
  (match (lcase-w c)
    [#f (length (lcase-y c))]
    [w (for/sum ([v (in-list w)] #:when (positive? v)) 1)]))

(define law-runs (make-hash))

(define (check-ran law got bound)
  (hash-update! law-runs law add1 0)
  (check-law law got bound))

(define (check-laws-ran laws)
  (for ([(law least) (in-hash laws)])
    (check-true (>= (hash-ref law-runs law 0) least)
                (format "~a ran ~a times, expected at least ~a" law (hash-ref law-runs law 0) least))))

(define (check-elastic-net-laws cases intercepts #:dtypes [dts dtypes] #:tols [ts tols])
  (hash-clear! law-runs)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (for* ([r (in-list l1-ratios)]
           [ratio (in-list alpha-ratios)])
      (define alpha (law-alpha p ratio r))
      (define o (oracle-elastic-net p alpha r))
      (match-define (struct* en-solution ([coef c*] [intercept b*] [mu mu] [mx mx])) o)
      (for* ([dt (in-list dts)]
             [tol (in-list ts)]
             [s (in-list selections)])
        (test-case (format "~a ~a l1-ratio ~a alpha ~a tol ~a ~a ~a"
                           (lcase-name c)
                           fi
                           r
                           ratio
                           tol
                           dt
                           s)
          (define u (unit-roundoff dt))
          (match-define (list coef b k)
            (en-case-fit c dt #:alpha alpha #:l1-ratio r #:fit-intercept fi #:tol tol #:selection s))
          (when (< k 1000)
            (match-define (kkt K B A) (host-kkt p coef alpha r tol k (rows-of c) u))
            (check-ran "S1" K (+ B A))
            (when (>= (max-abs coef) tol)
              (check-ran "S1 relative" K B))
            (when (> mu 0)
              (define x̄ (norm2 (map exact->inexact mx)))
              (define gap (norm2 (map (lambda (a e) (- (inexact->exact a) e)) coef c*)))
              (define bound (s2-bound o coef alpha r tol u))
              (check-ran "S2" gap bound)
              (when fi
                (check-ran "S3"
                           (abs (- (inexact->exact b) b*))
                           (+ (* x̄ bound)
                              (* 8 u (+ (abs b*) (* x̄ (norm2 (map exact->inexact c*))))))))))
          (check-true (exact-nonnegative-integer? k))))))
  (define fits (* (length cases) (length intercepts) (length l1-ratios) (length alpha-ratios)
                  (length dts) (length ts) (length selections)))
  (test-case "each tolerance law ran on at least half the fits"
    (check-laws-ran (hash "S1" (quotient fits 2)
                          "S1 relative" (quotient fits 2)
                          "S2" (quotient fits 2)
                          "S3" (if (memq #t intercepts) (quotient fits (* 2 (length intercepts))) 0)))))

(define (check-s2 c dt alpha r tol #:max-iter [max-iter 1000])
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define o (oracle-elastic-net p alpha r))
  (match-define (struct* en-solution ([coef c*] [mu mu])) o)
  (match-define (list coef _ k)
    (en-case-fit c dt #:alpha alpha #:l1-ratio r #:tol tol #:max-iter max-iter))
  (define u (unit-roundoff dt))
  (cond
    [(> mu 0)
     (check-law "S2" (distance coef c*) (s2-bound o coef alpha r tol u))
     "S2"]
    [else
     (match-define (kkt K B A) (host-kkt p coef alpha r tol k (rows-of c) u))
     (check-law "S1" K (+ B A))
     "S1"]))

(define (distance a b)
  (norm2 (map (lambda (x y) (exact->inexact (- (inexact->exact x) y))) a b)))
