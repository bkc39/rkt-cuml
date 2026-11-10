#lang racket/base

(require (only-in cuml/linear-model elastic-net-fit elastic-net-result lasso-result)
         (only-in racket/list append* append-map make-list)
         (only-in racket/match match match-define struct*)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in "blob-data.rkt" normal-source)
         (only-in "elastic-net-oracle.rkt"
                  alpha-max
                  en-solution
                  host-kkt
                  inexact-norm
                  kkt
                  norm2
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

(provide (struct-out en-run)
         alpha-ratios
         apply-keywords
         case-alpha
         check-accuracy
         check-elastic-net-laws
         coef-distance
         converged?
         correlated-case
         default-max-iter
         default-tol
         dyadic-weight-case
         en-case-fit
         en-fit
         keyword-hash
         l1-ratios
         law-alpha
         mirrored-weight-case
         selections
         tols)

(struct en-run (coef intercept n-iter) #:transparent)

(define default-max-iter 1000)
(define default-tol 1e-3)
(define intercept-law-rounding 8)

(define l1-ratios '(1 0.5 0.05 0))
(define alpha-ratios '(0.3 0.01 1e-4))
(define tols '(1e-2 1e-3 1e-5))
(define selections '(cyclic random))

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
  (match-define (struct* lcase ([X X] [y y])) (correlated-case 64 4 1/2 seed))
  (lcase (format "dyadic weights seed ~a" seed) X y (dyadic-weights)))

(define (mirrored xs negate)
  (append-map (lambda (x) (list x (negate x))) xs))

(define (mirrored-weight-case seed)
  (match-define (struct* lcase ([X X] [y y])) (correlated-case 32 4 1/2 seed))
  (lcase (format "mirrored dyadic weights seed ~a" seed)
         (mirrored X (lambda (row) (map - row)))
         (mirrored y -)
         (append-map (lambda (w) (list w w))
                     (for/list ([w (in-list (dyadic-weights))]
                                [i (in-naturals)]
                                #:when (even? i))
                       w))))

(define (en-fit r)
  (match-define (or (elastic-net-result c b k) (lasso-result c b k)) r)
  (en-run (device-vector->list c) b k))

(define (converged? run)
  (< (en-run-n-iter run) default-max-iter))

(define (keyword-hash kws vals)
  (for/hasheq ([k (in-list kws)] [v (in-list vals)])
    (values k v)))

(define (apply-keywords f options positional)
  (define kws (sort (hash-keys options) keyword<?))
  (keyword-apply f kws (map (lambda (k) (hash-ref options k)) kws) positional))

(define en-case-fit
  (make-keyword-procedure
   (lambda (kws vals c dt)
     (define given (keyword-hash kws vals))
     (en-fit (apply-keywords (hash-ref given '#:fit (lambda () elastic-net-fit))
                             (hash-set (hash-remove given '#:fit)
                                       '#:sample-weight
                                       (on-vector (lcase-w c) dt))
                             (list (on-matrix (lcase-X c) dt) (on-vector (lcase-y c) dt)))))))

(define (law-alpha p ratio l1-ratio)
  (* ratio (exact->inexact (alpha-max p (if (zero? l1-ratio) 1 l1-ratio)))))

(define (case-alpha c ratio l1-ratio #:fit-intercept [fi #t])
  (law-alpha (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi) ratio l1-ratio))

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
    (check-true
     (>= (hash-ref law-runs law 0) least)
     (format "~a ran ~a times, expected at least ~a" law (hash-ref law-runs law 0) least))))

(define (check-kkt-laws p run alpha r tol rows u)
  (match-define (struct* en-run ([coef coef] [n-iter k])) run)
  (match-define (kkt K B A) (host-kkt p coef alpha r tol k rows u))
  (check-ran "kkt-residual" K (+ B A))
  (when (>= (max-abs coef) tol)
    (check-ran "relative kkt-residual" K B)))

(define (check-distance-laws o run alpha r tol u fi)
  (match-define (struct* en-solution ([coef c*] [intercept b*] [mx mx])) o)
  (match-define (struct* en-run ([coef coef] [intercept b])) run)
  (define bound (s2-bound o coef alpha r tol u))
  (check-ran "coefficient-distance" (coef-distance coef c*) bound)
  (when fi
    (define x̄ (inexact-norm mx))
    (check-ran "intercept-distance"
               (abs (- (inexact->exact b) b*))
               (+ (* x̄ bound) (* intercept-law-rounding u (+ (abs b*) (* x̄ (inexact-norm c*))))))))

(define (check-elastic-net-laws cases intercepts #:dtypes [dts dtypes] #:tols [ts tols])
  (hash-clear! law-runs)
  (for* ([c (in-list cases)]
         [fi (in-list intercepts)])
    (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) fi))
    (for* ([r (in-list l1-ratios)]
           [ratio (in-list alpha-ratios)])
      (define alpha (law-alpha p ratio r))
      (define o (oracle-elastic-net p alpha r))
      (match-define (struct* en-solution ([mu mu])) o)
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
          (define run
            (en-case-fit c dt #:alpha alpha #:l1-ratio r #:fit-intercept fi #:tol tol #:selection s))
          (define checked (converged? run))
          (when checked
            (check-kkt-laws p run alpha r tol (rows-of c) u))
          (when (and checked (> mu 0))
            (check-distance-laws o run alpha r tol u fi))
          (check-true (exact-nonnegative-integer? (en-run-n-iter run)))))))
  (define fits
    (* (length cases)
       (length intercepts)
       (length l1-ratios)
       (length alpha-ratios)
       (length dts)
       (length ts)
       (length selections)))
  (test-case "each tolerance law ran on at least half the fits"
    (check-laws-ran (hash "kkt-residual" (quotient fits 2)
                          "relative kkt-residual" (quotient fits 2)
                          "coefficient-distance" (quotient fits 2)
                          "intercept-distance" (if (memq #t intercepts)
                                                   (quotient fits (* 2 (length intercepts)))
                                                   0)))))

(define (check-accuracy c dt alpha r tol
                        #:max-iter [max-iter default-max-iter]
                        #:law [law "coefficient-distance"])
  (define p (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t))
  (define o (oracle-elastic-net p alpha r))
  (match-define (struct* en-solution ([coef c*] [mu mu])) o)
  (match-define (struct* en-run ([coef coef] [n-iter k]))
    (en-case-fit c dt #:alpha alpha #:l1-ratio r #:tol tol #:max-iter max-iter))
  (define u (unit-roundoff dt))
  (check-equal? (if (> mu 0) "coefficient-distance" "kkt-residual") law "the law that applies")
  (match law
    ["coefficient-distance"
     (check-law law (coef-distance coef c*) (s2-bound o coef alpha r tol u))]
    ["kkt-residual"
     (match-define (kkt K B A) (host-kkt p coef alpha r tol k (rows-of c) u))
     (check-law law K (+ B A))]))

(define (coef-distance a b)
  (norm2 (map (lambda (x y) (exact->inexact (- (inexact->exact x) y))) a b)))
